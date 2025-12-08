import torch
import torch.distributed as dist

from .parallel_state import Group, get_tp_group, get_pp_group, get_world_group

from typing import Any, Union, Optional


def all_gather(
    input: torch.Tensor,
    dim: int = -1,
    group: Optional[Group] = None
) -> torch.Tensor:
    if dim < 0:
        # Convert negative dim to positive.
       dim += input.dim()
    group = get_world_group() if group is None else group
    input_size = input.size()
    world_size = group.size
    # Fast path: avoid invoking NCCL all_gather for singleton groups to
    # prevent potential driver/NCCL issues and improve performance.
    if world_size == 1:
        return input
    # NOTE: we have to use concat-style all-gather here,
    # stack-style all-gather has compatibility issues with
    # torch.compile . see https://github.com/pytorch/pytorch/issues/138795
    output_size = (input_size[0] * world_size, *input_size[1:])
    # Allocate output tensor.
    output_tensor = torch.empty(output_size, dtype=input.dtype, device=input.device)
    # All-gather.
    dist.all_gather_into_tensor(output_tensor, input, group=group.device_group)
    # Reshape
    output_tensor = output_tensor.reshape((world_size, ) + input_size)
    output_tensor = output_tensor.movedim(0, dim)
    output_tensor = output_tensor.reshape(input_size[:dim] +
                                            (world_size *
                                            input_size[dim], ) +
                                            input_size[dim + 1:])
    return output_tensor

def all_reduce(
    input: torch.Tensor,
    op = dist.ReduceOp.SUM,
    group: Optional[Group] = None
) -> torch.Tensor:
    group = get_world_group() if group is None else group
    world_size = group.size
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input
    dist.all_reduce(input, op, group=group.device_group)
    return input

def reduce_scatter(
    input: torch.Tensor,
    dim: int = -1,
    op = dist.ReduceOp.SUM,
    group: Optional[Group] = None
) -> torch.Tensor:
    group = get_world_group() if group is None else group
    world_size =  group.size
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input
    assert -input.dim() <= dim < input.dim(), (
        f"Invalid dim ({dim}) for input tensor with shape {input.size()}")

    if dim < 0:
        # Convert negative dim to positive.
        dim += input.dim()

    # Note: This will produce an incorrect answer if we don't make
    # the input_tensor contiguous. Possible bug in reduce_scatter_tensor?
    input_tensor = input.movedim(0, dim).contiguous()

    assert input_tensor.shape[0] % world_size == 0
    chunk_size = input_tensor.shape[0] // world_size
    output_shape = (chunk_size, ) + input_tensor.shape[1:]

    output_tensor = torch.empty(output_shape,
                                dtype=input_tensor.dtype,
                                device=input_tensor.device)

    # Perform reduce-scatter operation
    dist.reduce_scatter_tensor(output_tensor,
                                input_tensor,
                                group=group.device_group,
                                op=op)

    # Reshape before returning
    return output_tensor.movedim(0, dim).contiguous()

def gather(
    input: torch.Tensor,
    dst: int = 0,
    dim: int = -1,
    group: Optional[Group] = None
):
    """
    NOTE: We assume that the input tensor is on the same device across
    all the ranks.
    NOTE: `dst` is the local rank of the destination rank.
    """
    group = get_world_group() if group is None else group
    world_size = group.size
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input
    assert -input.dim() <= dim < input.dim(), (
        f"Invalid dim ({dim}) for input tensor with shape {input.size()}")
    if dim < 0:
        # Convert negative dim to positive.
        dim += input.dim()

    # Allocate output tensor.
    rank_in_group = group.group_rank
    if rank_in_group == dst:
        gather_list = [torch.empty_like(input) for _ in range(world_size)]
    else:
        gather_list = None
    # Gather.
    dist.gather(input,
                gather_list,
                group_dst=dst,
                group=group.device_group)
    if rank_in_group == dst:
        output_tensor = torch.cat(gather_list, dim=dim)
    else:
        output_tensor = None
    return output_tensor

def broadcast(
    input: torch.Tensor,
    src: int = 0,
    group: Optional[Group] = None
):
    """Broadcast the input tensor.
    NOTE: `src` is the local rank of the source rank.
    """
    group = get_world_group() if group is None else group
    world_size = group.size
    assert src < world_size, f"Invalid src rank ({src})"

    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input
    # Broadcast.
    dist.broadcast(
        input, group_src=src, group=group.device_group
    )
    return input

def tensor_model_parallel_all_reduce(input: torch.Tensor, op = dist.ReduceOp.SUM) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group."""
    tp_group = get_tp_group()
    return all_reduce(input, op, group=tp_group)

def tensor_model_parallel_all_gather(input: torch.Tensor,
                                     dim: int = -1) -> torch.Tensor:
    """All-gather the input tensor across model parallel group."""
    tp_group = get_tp_group()
    return all_gather(input, dim=dim, group=tp_group)

def tensor_model_parallel_reduce_scatter(input: torch.Tensor,
                                         dim: int = -1,
                                         op = dist.ReduceOp.SUM) -> torch.Tensor:
    """Reduce-Scatter the input tensor across model parallel group."""

    tp_group = get_tp_group()
    return reduce_scatter(input, dim=dim, op=op, group=tp_group)

def tensor_model_parallel_gather(input: torch.Tensor,
                                 dst: int = 0,
                                 dim: int = -1) -> torch.Tensor | None:
    """Gather the input tensor across model parallel group.
    NOTE: We assume that the input tensor is on the same device across
    all the ranks.
    NOTE: `dst` is the local rank of the destination rank.
    """
    tp_group = get_tp_group()
    return gather(input, dst=dst, dim=dim, group=tp_group)

def broadcast_tensor_dict(
    tensor_dict: dict[Any, torch.Tensor] | None = None,
    src: int = 0,
    group: Optional[Group] = None
) -> dict[Any, torch.Tensor] | None:
    """Broadcast the input tensor dictionary.
    NOTE: `src` is the local rank of the source rank.
    """
    group = get_world_group() if group is None else group
    
    world_size = group.size
    rank_in_group = group.group_rank
    # Bypass the function if we are using only 1 GPU.
    if (not dist.is_initialized() or world_size == 1):
        return tensor_dict

    assert src < world_size, f"Invalid src rank ({src})"

    metadata_list: list[tuple[Any, str, torch.dtype, torch.Size]] = []
    if rank_in_group == src:
        assert isinstance(tensor_dict, dict), f"Expecting a dictionary, got {type(tensor_dict)}"
        for key, tensor in tensor_dict.items():
            metadata_list.append((key, tensor.device.type, tensor.dtype, tensor.shape))

    # 先广播 metadata_list
    recv = [metadata_list]
    dist.broadcast_object_list(recv, group_src=src, group=group.cpu_group)
    metadata_list = recv[0]

    result: dict[Any, torch.Tensor] = {}
    async_handles: list[dist.Work] = []
    for key, device, dtype, shape in metadata_list:
        if rank_in_group == src:
            assert isinstance(tensor_dict, dict), f"Expecting a dictionary, got {type(tensor_dict)}"
            tensor = tensor_dict[key]
        else:
            tensor = torch.empty(shape, dtype=dtype, device=device)
            if tensor.numel() == 0:
                # Skip broadcasting empty tensors.
                result[key] = tensor
                continue
        if tensor.is_cpu:
            handle = dist.broadcast(tensor, group_src=src, group=group.cpu_group, async_op=True)  # CPU 张量广播
        else:
            handle = dist.broadcast(tensor, group_src=src, group=group.device_group, async_op=True)  # 直接张量广播
        if handle is not None:
            async_handles.append(handle)
        result[key] = tensor

    for async_handle in async_handles:
        async_handle.wait()
    return result

def send_tensor_dict(
    tensor_dict: dict[str, Union[torch.Tensor, Any]],
    dst: Optional[int] = None,
    group: Optional[Group] = None,
    all_gather_group: Optional[Group] = None,
):
    """Send the input tensor dictionary.
    NOTE: `dst` is the local rank of the source rank.
    """
    group = get_world_group() if group is None else group
    
    world_size = group.size
    rank_in_group = group.group_rank
    # Bypass the function if we are using only 1 GPU.
    if not dist.is_initialized() or world_size == 1:
        return

    all_gather_size = (1 if all_gather_group is None else
                        all_gather_group.size)
    all_gather_rank = (0 if all_gather_group is None else
                        all_gather_group.group_rank)

    if dst is None:
        dst = (rank_in_group + 1) % world_size
    assert dst < world_size, f"Invalid dst rank ({dst})"

    metadata_list: list[tuple[Any, str, torch.dtype, torch.Size]] = []
    tensor_list: list[torch.Tensor] = []
    assert isinstance(tensor_dict, dict), f"Expecting a dictionary, got {type(tensor_dict)}"
    for key, tensor in tensor_dict.items():
        metadata_list.append((key, tensor.device.type, tensor.dtype, tensor.shape))
        tensor_list.append(tensor)

    dist.send_object_list([metadata_list], group_dst=dst, group=group.cpu_group)
    
    for tensor in tensor_list:
        if tensor.numel() == 0:
            # Skip sending empty tensors.
            continue

        # send-allgather: send only a slice, then do allgather.
        if (all_gather_group is not None
                and tensor.numel() % all_gather_size == 0):
            tensor = tensor.reshape(all_gather_size, -1)[all_gather_rank]

        if tensor.is_cpu:
            # use cpu_group for CPU tensors
            dist.send(tensor, group_dst=dst, group=group.cpu_group)
        else:
            dist.send(tensor, group_dst=dst, group=group.device_group)

def recv_tensor_dict(
    src: Optional[int] = None,
    group: Optional[Group] = None,
    all_gather_group: Optional[Group] = None,
) -> Optional[dict[str, torch.Tensor]]:
    """Recv the input tensor dictionary.
    NOTE: `src` is the local rank of the source rank.
    """
    group = get_world_group() if group is None else group
    
    world_size = group.size
    rank_in_group = group.group_rank
    # Bypass the function if we are using only 1 GPU.
    if not dist.is_initialized() or world_size == 1:
        return None

    all_gather_size = (1 if all_gather_group is None else
                        all_gather_group.size)
    all_gather_rank = (0 if all_gather_group is None else
                        all_gather_group.group_rank)

    if src is None:
        src = (rank_in_group - 1) % world_size
    assert src < world_size, f"Invalid src rank ({src})"

    recv_metadata_list = [[]]
    dist.recv_object_list(recv_metadata_list, group_src=src, group=group.cpu_group)
    recv_metadata_list = recv_metadata_list[0]
    
    tensor_dict: dict[str, torch.Tensor] = {}
    for key, device, dtype, shape in recv_metadata_list:
        tensor = torch.empty(shape, dtype=dtype, device=device)
        if tensor.numel() == 0:
            # Skip broadcasting empty tensors.
            tensor_dict[key] = tensor
            continue

        # send-allgather: send only a slice, then do allgather.
        use_all_gather = (all_gather_group is not None
                            and tensor.numel() % all_gather_size == 0)

        orig_shape = tensor.shape
        if use_all_gather:
            tensor = tensor.reshape(all_gather_size,
                                    -1)[all_gather_rank]

        if tensor.is_cpu:
            # use cpu_group for CPU tensors
            dist.recv(tensor, group_src =src, group=group.cpu_group)
        else:
            # use group for GPU tensors
            dist.recv(tensor, group_src =src, group=group.device_group)
        if use_all_gather:
            # do the allgather
            tensor = all_gather(tensor, dim=0, group=all_gather_group)
            tensor = tensor.reshape(orig_shape)

        tensor_dict[key] = tensor

    return tensor_dict
