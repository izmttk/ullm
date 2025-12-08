import torch
from torch.distributed import ProcessGroup
import torch.distributed as dist
import datetime

class Group:
    def __init__(self, ranks: list[int], rank: int):
        self.ranks = ranks
        self.rank = rank
        self.size = len(ranks)
        # NOTE: all ranks must execute new_group when creating a new process group, even if they are not members of the group
        # NOTE: use_local_synchronization=True we can only call new_group for ranks in this group
        self.device_group: ProcessGroup = dist.new_group(ranks=ranks, use_local_synchronization=True)  # type: ignore
        self.cpu_group: ProcessGroup = dist.new_group(ranks=ranks, backend="gloo", use_local_synchronization=True)  # type: ignore
    
    @property
    def group_rank(self) -> int:
        return self.ranks.index(self.rank)
    
    @property
    def first_rank(self) -> int:
        return self.ranks[0]
    
    @property
    def last_rank(self) -> int:
        return self.ranks[-1]

    @property
    def is_first_rank(self) -> bool:
        return self.rank == self.first_rank
    
    @property
    def is_last_rank(self) -> bool:
        return self.rank == self.last_rank
    
    @property
    def prev_rank(self) -> int:
        group_rank = self.group_rank
        return self.ranks[(group_rank - 1) % self.size]

    @property
    def next_rank(self) -> int:
        group_rank = self.group_rank
        return self.ranks[(group_rank + 1) % self.size]

    def barrier(self):
        if self.size <= 1:
            return
        dist.barrier(self.cpu_group)
    
    def destroy(self):
        dist.destroy_process_group(self.device_group)
        del self.device_group
        dist.destroy_process_group(self.cpu_group)
        del self.cpu_group

WORLD: Group | None = None
TP: Group | None = None
PP: Group | None = None

# Attention: 所有的 rank 都指的是 global rank，除非变量或参数命名为 rank_in_group, tp_rank, pp_rank

def get_world_group() -> Group:
    assert WORLD is not None, "Distributed process group is not initialized."
    return WORLD

def get_tp_group() -> Group:
    """Get the tensor parallel process group."""
    assert TP is not None, "Tensor parallel process group is not initialized."
    return TP

def get_pp_group() -> Group:
    """Get the pipeline parallel process group."""
    assert PP is not None, "Pipeline parallel process group is not initialized."
    return PP

def initialize_model_parallel(tp_size: int, pp_size: int, tp_rank: int, pp_rank: int):
    global TP, PP

    tp_global_ranks = [pp_rank * tp_size + k for k in range(tp_size)]
    pp_global_ranks = [tp_rank + k * tp_size for k in range(pp_size)]
    
    TP = Group(ranks=tp_global_ranks, rank=tp_global_ranks[tp_rank])
    PP = Group(ranks=pp_global_ranks, rank=pp_global_ranks[pp_rank])

def destroy_model_parallel():
    global TP, PP
    if TP is not None:
        TP.destroy()
    if PP is not None:
        PP.destroy()
    TP = None
    PP = None

def init_distributed_environment(
    word_size: int = -1,
    rank: int = -1,
    backend: str = "nccl",
    init_method: str = "env://",
):
    global WORLD
    # print("Initializing distributed environment wtih rank", rank)
    if dist.is_initialized():
        print("Distributed process group is already initialized.")
        return
    # os.environ["NCCL_DEBUG"] = "INFO"
    # os.environ["NCCL_IB_DISABLE"] = "1"        # 禁用InfiniBand
    # os.environ["NCCL_P2P_DISABLE"] = "1"       # 禁用P2P通信
    # os.environ["NCCL_SHM_DISABLE"] = "1"       # 禁用共享内存
    # os.environ["NCCL_SOCKET_IFNAME"] = "eth0"  # 使用指定网络接口
    # os.environ["NCCL_PORT_RANGE"] = "50000-50100"  # 限制端口范围

    if backend == "nccl":
        # Set the device for each process based on its rank
        torch.cuda.set_device(rank)

    dist.init_process_group(
        backend=backend,
        init_method=init_method,
        world_size=word_size,
        rank=rank,
        timeout=datetime.timedelta(seconds=10),  # Set a timeout for the process group
        device_id=torch.device("cuda", rank) if backend == "nccl" else None,
    )
    # print(f"Distributed process group initialized. Backend: {backend}, World Size: {word_size}, Rank: {rank}")
    WORLD = Group(ranks=list(range(word_size)), rank=rank)

def destroy_distributed_environment():
    global WORLD
    if WORLD is not None:
        WORLD.destroy()
    WORLD = None
    
    if dist.is_initialized():
        dist.destroy_process_group()

def is_initialized() -> bool:
    return dist.is_initialized()
