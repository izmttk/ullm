from typing import Optional
from collections.abc import Generator
import torch
from torch import nn
from tqdm import tqdm
from safetensors import safe_open
import glob
import os
from .distributed.parallel_state import is_initialized, get_world_group

def default_weight_loader(param: torch.Tensor,
                          loaded_weight: torch.Tensor) -> None:
    """Default weight loader."""
    param.data.copy_(loaded_weight)

# explicitly use pure text format, with a newline at the end
# this makes it impossible to see the animation in the progress bar
# but will avoid messing up with ray or multiprocessing, which wraps
# each line of output with some prefix.
_BAR_FORMAT = "{desc}: {percentage:3.0f}% Completed | {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]\n"  # noqa: E501

def safetensors_weights_iterator(
    hf_weights_files: list[str]
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """
    Iterate over the weights in the model safetensor files.
    """
    enable_tqdm = (
        not is_initialized() or get_world_group().is_first_rank
    )
    for st_file in tqdm(
        hf_weights_files,
        desc="Loading safetensors checkpoint shards",
        disable=not enable_tqdm,
        bar_format=_BAR_FORMAT,
    ):
        with safe_open(st_file, framework="pt") as f:
            for name in f.keys():  # noqa: SIM118
                param = f.get_tensor(name)
                yield name, param

def load_model(model, hf_folder: str) -> nn.Module:

    hf_weights_files = glob.glob(os.path.join(hf_folder, "*.safetensors"))
    weights_iterator = safetensors_weights_iterator(hf_weights_files)
    # weights_list = list(safetensors_weights_iterator(hf_weights_files))
    model.load_weights(weights_iterator)
    return model.eval()
