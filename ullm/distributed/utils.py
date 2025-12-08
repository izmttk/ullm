def get_pp_indices(
    num_hidden_layers: int, pp_rank: int, pp_size: int
) -> tuple[int, int]:
    """
    Try to evenly distribute layers across partitions.

    If the number of layers is not divisible by the number of partitions,
    the last partition will have the remaining layers.
    """
    layers_per_partition = num_hidden_layers // pp_size
    start_layer = pp_rank * layers_per_partition
    end_layer = start_layer + layers_per_partition

    if pp_rank == pp_size - 1:
        end_layer = num_hidden_layers

    return (start_layer, end_layer)