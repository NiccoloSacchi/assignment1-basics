import torch
from torch import Tensor
from jaxtyping import Float, Int
from einops import einsum, reduce, rearrange



def cross_entropy_loss(
  inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
  # Since e^x can explode to inf for big x, subtract the maximum value for
  # numerical stability. Mathematically, this does not alter the result.
  max_inputs = inputs.max(dim=1, keepdim=True)[0]
  return (
    max_inputs +
    (inputs - max_inputs).exp().sum(dim=1, keepdim=True).log() -
    inputs[torch.arange(inputs.size(0)), targets]
  ).mean()
