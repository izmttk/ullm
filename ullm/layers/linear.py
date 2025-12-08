import torch
import torch.nn as nn
import torch.nn.functional as F
from ..distributed.parallel_state import get_tp_group
from ..distributed.communication_op import tensor_model_parallel_all_gather, tensor_model_parallel_all_reduce

from .utils import divide, ensure_divisibility

class ReplicatedLinear(nn.Module):
    """Replicated linear layer.

    Args:
        input_size: input dimension of the linear layer.
        output_size: output dimension of the linear layer.
        bias: If true, add bias.
        skip_bias_add: If true, skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
        quant_config: Quantization configure.
        prefix: The name of the layer in the state dict, including all parents
                        (e.g. model.layers.0.qkv_proj)
        return_bias: If true, return bias together with outputs in forward pass.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        return_bias: bool = True,
    ):

        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.skip_bias_add = skip_bias_add
        self.params_dtype = params_dtype
        self.return_bias = return_bias

        tp_size = get_tp_group().size
        tp_rank = get_tp_group().group_rank
        
        self.tp_size = tp_size
        self.tp_rank = tp_rank

        self.weight = nn.Parameter(
            torch.empty(
                self.output_size,
                self.input_size,
                dtype=params_dtype
            )
        )
        setattr(self.weight, "weight_loader", self.weight_loader)

        if bias:
            self.bias = nn.Parameter(
                torch.empty(
                    self.output_size,
                    dtype=self.params_dtype
                )
            )
            setattr(self.bias, "weight_loader", self.weight_loader)
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        assert param.shape == loaded_weight.shape, f"{param.shape=} != {loaded_weight.shape=}"
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        bias = self.bias if not self.skip_bias_add else None
        output = F.linear(x, self.weight, bias)
        output_bias = self.bias if self.skip_bias_add else None
        if not self.return_bias:
            return output
        return output, output_bias

class ColumnParallelLinear(nn.Module):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].
    
    before: Y = X A + b
    after: [Y_1, ..., Y_p] = [X A_1 + b_1, ..., X A_p + b_p]

    where:

    X is the input matrix of size (batch_size, input_size),
    A is the weight matrix of size (input_size, output_size),
    b is the bias vector of size (output_size,).
    Y is the output matrix of size (batch_size, output_size).
    p is the number of GPUs, and each GPU has a portion of the weight matrix.
    A_i is the weight matrix for the i-th GPU of size (input_size, output_size / p),
    Y_i is the output matrix for the i-th GPU of size (batch_size, output_size / p).
    
    Args:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias.
        gather_output: If true, call all-gather on output and make Y available
                       to all GPUs, otherwise, every GPU will have its output
                       which is Y_i = XA_i
        skip_bias_add: This was added to enable performance optimizations where
                       bias can be fused with other element-wise operations. we
                       skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
        return_bias: If true, return bias together with outputs in forward pass.
    """
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        gather_output: bool = True,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype  | None = None,
        return_bias: bool = True
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.gather_output = gather_output
        self.skip_bias_add = skip_bias_add
        self.params_dtype = params_dtype
        self.return_bias = return_bias

        tp_size = get_tp_group().size
        tp_rank = get_tp_group().group_rank
        
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.output_size_per_partition = divide(output_size, tp_size)
        
        self.weight = nn.Parameter(
            torch.empty(
                self.output_size_per_partition,
                self.input_size,
                dtype=params_dtype
            )
        )
        setattr(self.weight, "weight_loader", self.weight_loader)
        
        if bias:
            self.bias = nn.Parameter(
                torch.empty(
                    self.output_size_per_partition,
                    dtype=params_dtype
                )
            )
            setattr(self.bias, "weight_loader", self.weight_loader)
        else:
            self.register_parameter('bias', None)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        # 需注意，Linear 的 weight 形状为 (output_size, input_size)
        output_dim = 0
        param_data = param.data
        shard_size = param_data.shape[output_dim]
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)
        assert param_data.shape == loaded_weight.shape, f"{param_data.shape=} != {loaded_weight.shape=}"
        param.data.copy_(loaded_weight)
            
    def forward(self, x: torch.Tensor):
        bias = self.bias if not self.skip_bias_add else None
        output_parallel = F.linear(x, self.weight, bias)
        if self.gather_output:
            output = tensor_model_parallel_all_gather(output_parallel)
        else:
            output = output_parallel
        output_bias = self.bias if self.skip_bias_add else None
        if not self.return_bias:
            return output
        return output, output_bias

class RowParallelLinear(nn.Module):
    """Linear layer with row parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its first dimension and X along its second dimension as:
               -   -
              | A_1 |
              | .   |
          A = | .   |        X = [X_1, ..., X_p]
              | .   |
              | A_p |
               -   -

    Y = X_1 A_1 + ... + X_pA_p + b

    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias. Note that bias is not parallelized.
        input_is_parallel: If true, we assume that the input is already
                           split across the GPUs and we do not split
                           again.
        skip_bias_add: This was added to enable performance optimization where
                       bias can be fused with other element-wise operations.
                       We skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
        reduce_results: If true, call all-reduce on output and make Y available
                       to all GPUs, otherwise, every GPU will have its output
                       which is Y = X_iA_i
        return_bias: If true, return bias together with outputs in forward pass.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        input_is_parallel: bool = True,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        reduce_results: bool = True,
        return_bias: bool = True
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.input_is_parallel = input_is_parallel
        self.skip_bias_add = skip_bias_add
        self.params_dtype = params_dtype
        self.reduce_results = reduce_results
        self.return_bias = return_bias

        tp_size = get_tp_group().size
        tp_rank = get_tp_group().group_rank
        self.tp_size = tp_size
        self.tp_rank = tp_rank

        self.input_size_per_partition = divide(input_size, tp_size)
        self.weight = nn.Parameter(
            torch.empty(
                self.output_size,
                self.input_size_per_partition,
                dtype=params_dtype
            )
        )
        setattr(self.weight, "weight_loader", self.weight_loader)
        if bias:
            self.bias = nn.Parameter(
                torch.empty(
                    self.output_size,
                    dtype=params_dtype
                )
            )
            setattr(self.bias, "weight_loader", self.weight_loader)
        else:
            self.register_parameter('bias', None)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        # 需注意，Linear 的 weight 形状为 (output_size, input_size)
        input_dim = 1
        param_data = param.data
        shard_size = param_data.shape[input_dim]
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(input_dim, start_idx, shard_size)
        assert param_data.shape == loaded_weight.shape, f"{param_data.shape=} != {loaded_weight.shape=}"
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        if self.input_is_parallel:
            input_parallel = x
        else:
            splitted_input = torch.split(x, self.input_size_per_partition, dim=-1)
            input_parallel = splitted_input[self.tp_rank].contiguous()

        # Only fuse bias add into GEMM for rank 0 (this ensures that
        # bias will not get added more than once in TP>1 case)
        bias = None if (self.tp_rank > 0 or self.skip_bias_add) else self.bias
        output_parallel = F.linear(input_parallel, self.weight, bias)
        if self.reduce_results and self.tp_size > 1:
            output = tensor_model_parallel_all_reduce(output_parallel)
        else:
            output = output_parallel

        output_bias = self.bias if self.skip_bias_add else None

        if not self.return_bias:
            return output
        return output, output_bias


class QKVParallelLinear(ColumnParallelLinear):
    """Linear layers for the attention's QKV transformation.

    Linear layers for the linear transformation of the query, key, and value
    vectors in the attention layer. The weight matrix is concatenated along
    the output dimension. The layer is parallelized along the head dimension.
    When the number of key/value heads is smaller than the number of query
    heads (e.g., multi-query/grouped-query attention), the key/value head may
    be replicated while the query heads are partitioned.

    Args:
        hidden_size: input hidden state size of the transformer.
        head_size: size of each attention head.
        total_num_heads: total number of attention query heads.
        total_num_kv_heads: total number of attention key/value heads. If
                            None, assume total_num_kv_heads = total_num_heads.
        bias: If true, add bias.
        skip_bias_add: This was added to enable performance optimizations where
                       bias can be fused with other element-wise operations. we
                       skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
        return_bias: If true, return bias together with outputs in forward pass.
    """

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = True,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype |  None = None,
        return_bias: bool = True
    ):
        self.hidden_size = hidden_size
        self.head_size = head_size
        self.total_num_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads if total_num_kv_heads is not None else total_num_heads
        self.params_dtype = params_dtype

        tp_size = get_tp_group().size
        self.num_heads = divide(total_num_heads, tp_size)

        if tp_size > self.total_num_kv_heads:
            self.num_kv_heads = 1
            self.num_kv_head_replicas = divide(tp_size, self.total_num_kv_heads)
        else:
            self.num_kv_heads = divide(self.total_num_kv_heads, tp_size)
            self.num_kv_head_replicas = 1

        input_size = hidden_size
        output_size = (self.num_heads + 2 * self.num_kv_heads) * tp_size * self.head_size

        super().__init__(
            input_size=input_size,
            output_size=output_size,
            bias=bias,
            gather_output=False,
            skip_bias_add=skip_bias_add,
            params_dtype=params_dtype,
            return_bias=return_bias
        )

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: str | None = None
    ):
        # 需注意，Linear 的 weight 形状为 (output_size, input_size)
        output_dim = 0
        param_data = param.data

        if loaded_shard_id is None:
            shard_offsets = [
                # (shard_id, shard_offset, shard_size)
                (
                    "q", 
                    0,
                    self.total_num_heads * self.head_size
                ),
                (
                    "k",
                    self.total_num_heads * self.head_size,
                    self.total_num_kv_heads * self.head_size,
                ),
                (
                    "v",
                    (self.total_num_heads + self.total_num_kv_heads) * self.head_size,
                    self.total_num_kv_heads * self.head_size,
                ),
            ]
            for shard_id, shard_offset, shard_size in shard_offsets:
                loaded_weight_shard = loaded_weight.narrow(output_dim, shard_offset, shard_size)
                self.weight_loader(param, loaded_weight_shard, shard_id)
            return

        if loaded_shard_id == "q":
            shard_offset = 0
            shard_size = self.num_heads * self.head_size
        elif loaded_shard_id == "k":
            shard_offset = self.num_heads * self.head_size
            shard_size = self.num_kv_heads * self.head_size
        elif loaded_shard_id == "v":
            shard_offset = (self.num_heads + self.num_kv_heads) * self.head_size
            shard_size = self.num_kv_heads * self.head_size
        else:
            assert False, f"Invalid shard_id {loaded_shard_id}"

        param_data = param_data.narrow(output_dim, shard_offset, shard_size)
        if loaded_shard_id == "q":
            shard_id = self.tp_rank
        else:
            shard_id = self.tp_rank // self.num_kv_head_replicas
        start_idx = shard_id * shard_size
        loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)

        assert param_data.shape == loaded_weight.shape, f"{param_data.shape=} != {loaded_weight.shape=}"
        param_data.copy_(loaded_weight)

class MergedColumnParallelLinear(ColumnParallelLinear):
    """Packed linear layers with column parallelism.

    Similar to ColumnParallelLinear, but the weight matrix is concatenated
    along the output dimension. When the weight matrix is loaded, the
    different partitions are sharded separately.

    Y = X [A1, A2, ..., An] + b
    each Ai is separated into p shards along the output dimension as Ai = [Ai_1, Ai_2, ..., Ai_p].

    for rank 0: Y_1 = X [A1_1, A2_1, ..., An_1] + b_1
    for rank 1: Y_2 = X [A1_2, A2_2, ..., An_2] + b_2
    ...
    for rank p: Y_p = X [A1_p, A2_p, ..., An_p] + b_p

    Args:
        input_size: input dimension of the linear layer.
        output_sizes: list of output dimensions of the linear layer.
        bias: If true, add bias.
        gather_output: If true, call all-gather on output and make the output
                       available to all GPUs, otherwise, every GPU will have
                       its own output.
        skip_bias_add: This was added to enable performance optimizations where
                       bias can be fused with other element-wise operations. we
                       skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
        quant_config: Quantization configure.
        prefix: The name of the layer in the state dict, including all parents
                        (e.g. model.layers.0.qkv_proj)
    """

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = True,
        gather_output: bool = False,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
    ):
        self.output_sizes = output_sizes

        tp_size = get_tp_group().size

        for output_size in output_sizes:
            ensure_divisibility(output_size, tp_size)

        super().__init__(
            input_size=input_size,
            output_size=sum(output_sizes),
            bias=bias,
            gather_output=gather_output,
            skip_bias_add=skip_bias_add,
            params_dtype=params_dtype,
        )
    
    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: int | None = None
    ):
        # 需注意，Linear 的 weight 形状为 (output_size, input_size)
        output_dim = 0
        param_data = param.data

        if loaded_shard_id is None:
            current_shard_offset = 0
            shard_offsets: list[tuple[int, int, int]] = []
            for i, output_size in enumerate(self.output_sizes):
                # (shard_id, shard_offset, shard_size)
                shard_offsets.append((i, current_shard_offset, output_size))
                current_shard_offset += output_size
            for shard_id, shard_offset, shard_size in shard_offsets:
                loaded_weight_shard = loaded_weight.narrow(output_dim, shard_offset, shard_size)
                self.weight_loader(param, loaded_weight_shard, shard_id)
            return

        assert loaded_shard_id < len(self.output_sizes), f"Invalid shard_id {loaded_shard_id}"
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size

        param_data = param_data.narrow(output_dim, shard_offset, shard_size)
        start_idx = self.tp_rank * shard_size

        loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)

        assert param_data.shape == loaded_weight.shape, f"{param_data.shape=} != {loaded_weight.shape=}"
        param_data.copy_(loaded_weight)
