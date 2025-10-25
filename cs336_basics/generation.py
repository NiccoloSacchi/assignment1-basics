import torch
from torch import nn, Tensor
import numpy as np
from einops import rearrange
from jaxtyping import Int

from cs336_basics.utils import Tokenizer, softmax


def generate_text(
    model: nn.Module,
    tokenizer: Tokenizer,
    input_text: str,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = 0.0,
) -> str:
    """
    Generate new tokens from the model given the input token ids.

    Args:
        model: The language model to use for generation.
        tokenizer: The tokenizer to use for encoding and decoding.
        input_ids: Tensor of shape (batch_size, context_length) containing the
          input token ids.
        max_new_tokens: The maximum number of new tokens to generate.
        temperature: The temperature to use for sampling. Higher values result
          in more random samples.
        top_p: If > 0, keep only the smallest set of tokens whose cumulative
          probability mass exceeds top_p.

    Returns:
        A tensor of length up to max_new_tokens containing the input_ids with
        the generated tokens appended.
    """
    eos_token = tokenizer.encode("<|endoftext|>")
    if len(eos_token) != 1:
      eos_token = None

    input_tokens = tokenizer.encode(input_text)
    tokens = generate_tokens(
        model,
        torch.tensor(input_tokens, dtype=torch.int32),
        max_new_tokens,
        temperature,
        top_p,
        eos_token_id=eos_token[0] if eos_token is not None else None,
    )
    return tokenizer.decode(tokens.tolist())


def generate_tokens(
    model: nn.Module,
    input_tokens: Int[Tensor, "seq_length"],
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = 0.0,
    eos_token_id: int | None = None,
) -> Int[Tensor, "new_seq_length"]:
    """
    Generate new tokens from the model given the input token ids.

    Args:
        model: The language model to use for generation.
        input_ids: Tensor of shape (batch_size, context_length) containing the
          input token ids.
        max_new_tokens: The maximum number of new tokens to generate.
        temperature: The temperature to use for sampling. Higher values result
          in more random samples.
        top_p: If > 0, keep only the smallest set of tokens whose cumulative
          probability mass exceeds top_p.

    Returns:
        A tensor of length up to max_new_tokens containing the input_ids with
        the generated tokens appended.
    """
    tokens = rearrange(input_tokens, 'seq_length -> 1 seq_length')
    with torch.no_grad():
      for _ in range(max_new_tokens):
        logits = model(tokens)
        probs = softmax(logits[0, -1, :] / temperature, dim=-1)
        next_token = np.random.choice(len(probs), p=probs.cpu().numpy())
        tokens = torch.cat([tokens, torch.tensor([[next_token]], dtype=tokens.dtype)], dim=1)
        if eos_token_id is not None and next_token == eos_token_id:
          break
    return tokens[0]