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
        num_checkpoints: int = 0,
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
          num_checkpoints: number of checkpoints for the num_layers. Defaults to
            0, meaning no checkpointing is done.
        """
        super().__init__()
        assert (
            num_layers >= num_checkpoints
        ), "You cannot have more checkpoints than TransformerBlock layers."

        # Used to save and load how the model.
        self._init_args = {
            "vocab_size": vocab_size,
            "context_length": context_length,
            "num_layers": num_layers,
            "d_model": d_model,
            "num_heads": num_heads,
            "d_ff": d_ff,
            "rope_theta": rope_theta,
            "rms_norm_eps": rms_norm_eps,
            "device": device,
            "dtype": dtype,
            "num_checkpoints": num_checkpoints,
        }
        self.num_checkpoints = num_checkpoints
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
        self.rms = RMSNorm(d_model, rms_norm_eps, device, dtype)
        self.linear = Linear(d_model, vocab_size, device, dtype)

    def init_args(self):
        return self._init_args

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
        if self.num_checkpoints > 0:
            layers_per_checkpoint = self.num_layers // self.num_checkpoints
            for i in range(0, self.num_layers, layers_per_checkpoint):
                segment = self.transformer_blocks[i : i + layers_per_checkpoint]

                def run_segment(start_x, layers=segment):
                    for layer in layers:
                        start_x = layer(start_x)
                    return start_x

                # use_reentrant=False is recommended for torch.compile compatibility.
                x = checkpoint(
                    run_segment, x, use_reentrant=False
                )  # (..., context_length, d_model)
        else:
            # Standard execution
            for block in self.transformer_blocks:
                x = block(x)  # (..., context_length, d_model)

        x = self.rms(x)  # (..., context_length, d_model)
        logits = self.linear(x)  # (..., context_length, vocab_size)
        return logits  # (..., context_length, vocab_size)
