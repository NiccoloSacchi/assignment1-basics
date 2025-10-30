import math
import torch
from torch import Tensor
from jaxtyping import Float, Int
from einops import rearrange


def cross_entropy_loss(
  inputs: Float[Tensor, "batch_dimensions vocab_size"],
  targets: Int[Tensor, "batch_dimensions"],
) -> Float[Tensor, ""]:
  # Since e^x can explode to inf for big x, subtract the maximum value for
  # numerical stability. Mathematically, this does not alter the result.
  max_inputs = inputs.max(dim=-1, keepdim=True)[0]

  targets_unsqueezed = rearrange(targets, "... -> ... 1")
  target_inputs = torch.gather(inputs, -1,  targets_unsqueezed.long())
  return (
    max_inputs +
    (inputs - max_inputs).exp().sum(dim=-1, keepdim=True).log() -
    target_inputs
  ).mean()


def perplexity(loss: float) -> float:
    """Convert cross-entropy loss to perplexity."""
    return math.exp(loss)
