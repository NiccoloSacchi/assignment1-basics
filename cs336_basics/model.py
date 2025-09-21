import torch
from torch import nn, Tensor
from jaxtyping import Float, Int
from cs336_basics.layers import (
  Embedding,
  TransformerBlock,
  RMSNorm,
  Linear,
)  

class TransformerLM(nn.Module):
  def __init__(
      self,
      vocab_size: int,
      context_length: int,
      num_layers: int,
      d_model: int,
      num_heads: int,
      d_ff: int,
      rope_theta: float | None = None,
      rms_norm_eps: float = 1e-5,
      device: torch.device = torch.device("cpu"),
      dtype: torch.dtype = torch.float32,
  ):
    """
    Args:
      vocab_size: The size of the vocabulary, necessary for determining
        the dimensionality of the token embedding matrix.
      context_length: The maximum context length, necessary for
        determining the dimensionality of the position embedding matrix.
      num_layers: The number of Transformer blocks to use.
      d_model: Hidden dimension of the model and of the input of the Transformer
        block.
      num_heads: Number of attention heads.
      d_ff: Dimensionality of the position-wise feed-forward inner layer.
      rope_theta: If not None, use RoPE with the given base value to compute
        the rotation angles.
      rms_norm_eps: Epsilon value for numerical stability in RMSNorm.
      device: Device to store the parameters on.
      dtype: Data type of the parameters.
    """
    super().__init__()
    self.token_embeddings = Embedding(vocab_size, d_model, device, dtype)
    self.transformer_blocks = nn.ModuleList(
      [
        TransformerBlock(
          d_model,
          num_heads,
          d_ff,
          context_length,
          rope_theta,
          rms_norm_eps,
          device,
          dtype,
        )
        for _ in range(num_layers)
      ]
    )
    self.norm = RMSNorm(d_model, rms_norm_eps, device, dtype)
    self.linear = Linear(d_model, vocab_size, device, dtype)

  def forward(
      self,
      input_ids: Int[Tensor, " ... context_length"],
    ) -> Float[Tensor, " ... context_length vocab_size"]:
      """
      Args:
        input_ids: Tensor of shape (..., context_length) containing the token
          ids.
      Returns:
        logits: Tensor of shape (..., context_length, vocab_size) containing
          the logits for each token in the vocabulary.
      """
      x = self.token_embeddings(input_ids)  # (..., context_length, d_model)
      for block in self.transformer_blocks:
        x = block(x)  # (..., context_length, d_model)
      x = self.norm(x)  # (..., context_length, d_model)
      logits = self.linear(x)  # (..., context_length, vocab_size)
      return logits  # (..., context_length, vocab_size)

