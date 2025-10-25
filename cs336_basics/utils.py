"""Utility functions."""

from typing import Protocol
from jaxtyping import Float
import torch
from torch import Tensor
import tqdm
import pathlib
import numpy as np


def get_tqdm(iterable, condition=True, **kwargs):
  if condition:
    return tqdm.tqdm(iterable, **kwargs)
  return iterable


def concatenate_documents(data: list[str], special_token="<|endoftext|>"):
  return special_token.join(data)


class Tokenizer(Protocol):
    def encode(self, text: str) -> list[int]:
        ...
    def decode(self, tokens: list[int], verbose: bool = False) -> str:
        ...

def compression_ratio(tokenizer: Tokenizer, text: str) -> float:
  """Computes the compression ratio (bytes/token) of a tokenizer on a given text.

  Args:
    tokenizer: The tokenizer to use.
    text: The text to encode.

  Returns:
    The compression ratio (bytes/token).
  """
  num_bytes = len(text.encode('utf-8'))
  num_tokens = len(tokenizer.encode(text))
  return num_bytes/num_tokens if num_tokens > 0 else float('inf')

def get_batch(
    dataset: np.typing.NDArray,
    batch_size: int,
    context_length: int,
    device: str,
    dtype: torch.dtype | None = None,
) -> tuple[
  Float[Tensor, "batch_size context_length"],
  Float[Tensor, "batch_size context_length"],
]:
    """
    Sample random batches from a dataset for language modeling.
    
    Args:
        dataset: 1D numpy array of integer token IDs. Also works with np.memmap.
        batch_size: Number of sequences to sample.
        context_length: Length of each sequence.
        device: PyTorch device string (e.g., 'cpu' or 'cuda:0').
        dtype: Target dtype for the returned tensors. If specified, tensors are
          converted to this dtype before moving to device.
    
    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length).
        First tensor contains input sequences, second contains target sequences 
        (input shifted by 1).
    """
    assert context_length < len(dataset), "Context length must be less than dataset length."
    
    # Sample random starting indices.
    indices = np.random.choice(len(dataset) - context_length, batch_size, replace=False)
    
    # Create input sequences (x) and target sequences (y, shifted by 1).
    x_np = np.empty((batch_size, context_length), dtype=dataset.dtype)
    y_np = np.empty((batch_size, context_length), dtype=dataset.dtype)
    for idx, i in enumerate(indices):
        x_np[idx] = dataset[i:i+context_length]
        y_np[idx] = dataset[i+1:i+1+context_length]
    x = torch.from_numpy(x_np)
    y = torch.from_numpy(y_np)

    if dtype:
      x = x.to(dtype)
      y = y.to(dtype)

    # Move to specified device.
    x = x.to(device)
    y = y.to(device)
    return x, y


def softmax(
  x: Float[Tensor, "... D"],
  dim: int,
) -> Float[Tensor, "... D"]:
  """Numerically stable softmax implementation.
  
  Args:
    x: Input tensor.
    dim: dimenstion along which to apply the softmax operation.

  Returns:
    Tensor of same shape as input with softmax applied on the last dimension.
  """
  # Since e^x can explode to inf for big x, subtract the maximum value for
  # numerical stability. Mathematically, this does not alter the result.
  x_exp = torch.exp((x - x.max(dim=dim, keepdim=True)[0]))
  return x_exp / x_exp.sum(dim=dim, keepdim=True)
