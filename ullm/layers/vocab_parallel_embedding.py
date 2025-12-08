import torch
from torch import nn
import torch.nn.functional as F

from ..distributed.parallel_state import get_tp_group, get_pp_group
from ..distributed.communication_op import tensor_model_parallel_all_reduce, tensor_model_parallel_all_gather

from .utils import divide

DEFAULT_VOCAB_PADDING_SIZE = 64

def pad_vocab_size(vocab_size: int,
                   pad_to: int = DEFAULT_VOCAB_PADDING_SIZE) -> int:
    """Pad the vocab size to the given value."""
    return ((vocab_size + pad_to - 1) // pad_to) * pad_to

class VocabParallelEmbedding(torch.nn.Module):
    """Embedding parallelized in the vocabulary dimension.

    Adapted from torch.nn.Embedding, note that we pad the vocabulary size to
    make sure it is divisible by the number of model parallel GPUs.

    In this example, we will have the vocab size = 1010, and padding to 64. 
    Therefore, the total vocab size with padding will be 1024.
    Therefore, the tensor format looks like the following:
    TP1, rank 0 (no sharding):
                            |< --------BASE-------- >|< -BASE PADDING-- >|
    corresponding token_id: |  0  |  1  | ... | 1009 |  -1  | ... |  -1  |
                     index: |  0  |  1  | ... | 1009 | 1010 | ... | 1023 |

    TP2, rank 0:
                            |< --------------------BASE--------------------- >|
    corresponding token_id: |  0  |  1  |  2  | ... | 497  | 498 | ...  | 511 |
                     index: |  0  |  1  |  2  | ... | 497  | 498 | ...  | 511 |
    TP2, rank 1:
                            |< -----------BASE----------- >|< -BASE PADDING- >|
    corresponding token_id: | 512 | 513 | 514 | ... | 1009 | -1  | ...  | -1  |
                     index: |  0  |  1  |  2  | ... | 497  | 498 | ...  | 511 |

    Args:
        num_embeddings: vocabulary size.
        embedding_dim: size of hidden state.
        params_dtype: type of the parameters.
        padding_size: padding size for the vocabulary.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        params_dtype: torch.dtype | None = None,
        padding_size: int = DEFAULT_VOCAB_PADDING_SIZE,
    ):
        super().__init__()

        self.tp_rank = get_tp_group().group_rank
        self.tp_size = get_tp_group().size

        self.num_embeddings = num_embeddings
        self.padding_size = padding_size

        self.num_embeddings_padded = pad_vocab_size(
            self.num_embeddings, self.padding_size
        )
        self.embedding_dim = embedding_dim

        # Divide the weight matrix along the vocaburaly dimension.
        self.num_embeddings_per_partition = divide(self.num_embeddings_padded, self.tp_size)

        self.vocab_start_index = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_index = self.vocab_start_index + self.num_embeddings_per_partition
        
        self.weight = nn.Parameter(
            torch.empty(
                self.num_embeddings_per_partition, self.embedding_dim, dtype=params_dtype
            )
        )
        setattr(self.weight, "weight_loader", self.weight_loader)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):

        # Shard indexes for loading the weight
        start_idx = self.vocab_start_index
        shard_size = self.vocab_end_index - start_idx

        assert loaded_weight.shape[0] == self.num_embeddings, f"{self.num_embeddings=} {loaded_weight.shape[0]=}"

        # Copy the data.
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        param[: loaded_weight.shape[0]].data.copy_(loaded_weight)
        param[loaded_weight.shape[0] :].data.fill_(0)

    def forward(self, x: torch.Tensor):
        if self.tp_size > 1:
            # Build the mask.
            input_mask = (x >= self.vocab_start_index) & (x < self.vocab_end_index)
            # Apply the mask.
            x = x - self.vocab_start_index
            x.masked_fill_(~input_mask, 0)
            # Get the embeddings.
            output_parallel = F.embedding(x.long(), self.weight)
            # Mask the output embedding.
            output_parallel.masked_fill_(~input_mask.unsqueeze(-1), 0)
            # Reduce across all the model parallel GPUs.
            output = tensor_model_parallel_all_reduce(output_parallel)
        else:
            output = F.embedding(x, self.weight)
        return output

class ParallelLMHead(VocabParallelEmbedding):
    """Parallelized LM head.

    Output logits weight matrices used in the Sampler. The weight and bias
    tensors are padded to make sure they are divisible by the number of
    model parallel GPUs.

    Args:
        num_embeddings: vocabulary size.
        embedding_dim: size of hidden state.
        bias: whether to use bias.
        params_dtype: type of the parameters.
        padding_size: padding size for the vocabulary.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
        params_dtype: torch.dtype | None = None,
        padding_size: int = DEFAULT_VOCAB_PADDING_SIZE,
    ):
        super().__init__(
            num_embeddings,
            embedding_dim,
            params_dtype=params_dtype,
            padding_size=padding_size,
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(self.num_embeddings_per_partition, dtype=params_dtype)
            )
            setattr(self.bias, "weight_loader", self.weight_loader)
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor):
        logits = F.linear(x, self.weight, self.bias)
        logits = tensor_model_parallel_all_gather(logits)
        logits = logits[:, : self.num_embeddings]
        return logits
