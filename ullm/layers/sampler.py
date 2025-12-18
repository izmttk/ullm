import torch
from torch import nn

class Sampler(nn.Module):
    """Samples the next tokens from the model's outputs.

    This layer does the following:
    1. Discard the hidden states that are not used for sampling (i.e., all
        tokens except the final one in each prompt).
    2. Compute the logits for the next tokens.
    4. Apply temperature scaling.
    5. Apply top-p and top-k truncation.
    6. Sample the next tokens.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor | None = None,
        min_ps: torch.Tensor | None = None,
        top_ps: torch.Tensor | None = None,
        top_ks: torch.Tensor | None = None,
    ):
        is_all_greedy = (
            temperatures is None and
            min_ps is None and
            top_ps is None and
            top_ks is None
        )

        if is_all_greedy:
            return torch.argmax(logits, dim=-1)
        # Use float32 to apply temperature scaling.
        # Use in-place division to avoid creating a new tensor.
        logits = logits.to(torch.float)
        if temperatures is not None:
            logits.div_(temperatures.unsqueeze(dim=1))

        logits = _apply_top_k_top_p(logits, top_ps, top_ks)
        logits = _apply_min_p(logits, min_ps)

        # We use float32 for probabilities and log probabilities.
        # Compute the probabilities.
        probs = torch.softmax(logits, dim=-1, dtype=torch.float)
        # Compute the log probabilities.
        # logprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float)

        # Sample the next tokens.
        batch_next_token_ids = torch.multinomial(probs, num_samples=1)

        return batch_next_token_ids.squeeze(dim=-1)


def _apply_top_k_top_p(
    logits: torch.Tensor,
    p: torch.Tensor | None = None,
    k: torch.Tensor | None = None
) -> torch.Tensor:
    if p is None and k is None:
        return logits

    logits_sort, logits_idx = logits.sort(dim=-1, descending=False)

    if k is not None:
        # Apply top-k.
        top_k_mask = logits_sort.size(1) - k.to(torch.long)
        # Get all the top_k values.
        top_k_mask = logits_sort.gather(1, top_k_mask.unsqueeze(dim=1))
        top_k_mask = logits_sort < top_k_mask
        logits_sort.masked_fill_(top_k_mask, -float("inf"))


    if p is not None:
        # Apply top-p.
        probs_sort = logits_sort.softmax(dim=-1)
        probs_sum = probs_sort.cumsum(dim=-1)
        top_p_mask = probs_sum <= 1 - p.unsqueeze(dim=1)
        # at least one
        top_p_mask[:, -1] = False
        logits_sort.masked_fill_(top_p_mask, -float("inf"))

    # Re-sort the probabilities.
    logits = torch.empty_like(logits_sort).scatter_(dim=-1,
                                                    index=logits_idx,
                                                    src=logits_sort)
    return logits


def _apply_min_p(
    logits: torch.Tensor,
    min_p: torch.Tensor | None = None
) -> torch.Tensor:
    if min_p is None:
        return logits

    probs = torch.softmax(logits, dim=-1)
    top_probs, _ = probs.max(dim=-1, keepdim=True)
    scaled_min_p = min_p.unsqueeze_(dim=1) * top_probs
    tokens_to_remove = probs < scaled_min_p
    logits = logits.masked_fill_(tokens_to_remove, -float("inf"))

    return logits
