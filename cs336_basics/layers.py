import math
import torch
from torch import nn
from torch import Tensor
from jaxtyping import Float, Int, Bool
from einops import einsum, reduce, rearrange


def xavier_initialized_tensor(
  shape: tuple[int],
  in_features: int,
  out_features: int,
  device: torch.device | None = None,
  dtype: torch.dtype | None = None,
):
  """Returns a Normal Xavier-initialized tensor of the given shape.  
  """
  std = (2 / (in_features + out_features)) ** 0.5
  return torch.nn.init.trunc_normal_(
      torch.empty(shape, device=device, dtype=dtype),
      mean=0,
      std=std,
      a=-3*std,
      b=3*std,
  )


class Linear(nn.Module):
  """A linear layer (like nn.Linear) with Xavier initialization.
  """
  def __init__(
    self,
    in_features: int,
    out_features: int,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None
):
    """
    Args:
      in_features: final dimension of the input.
      out_features: final dimension of the output.
      device: Device to store the parameters on.
      dtype: Data type of the parameters.
    """
    super().__init__()
    self.in_features = in_features
    self.out_features = out_features
    # Store as (out_features, in_features) for memory ordering reasons: PyTorch
    # stores multi-dimensional arrays in row-major order and when multiplying
    # the input with w we need read the in_features.
    self.w = nn.Parameter(
        xavier_initialized_tensor(
          [out_features, in_features],
          in_features,
          out_features,
          device,
          dtype,
        )
    )
  
  def forward(self, x: Float[Tensor, " ... in_features"]) -> torch.Tensor:
    return einsum(self.w, x, "out_features in_features, ... in_features -> ... out_features")
  
  def set_weights(self, weight: Float[Tensor, " out_features in_features"]):
    """Set the weights of the layer.
    
    Args:
      weight: Weights for the linear layer.
    """
    assert weight.shape == self.w.data.shape, "unexpected shape for weight: got {}, expected {}".format(weight.shape, self.w.data.shape)
    self.w.data = weight


class Embedding(nn.Module):
  """Embedding layer for mapping token IDs to dense vectors (like nn.Embedding).
  """
  def __init__(
    self,
    num_embeddings: int,
    embedding_dim: int,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None
):
    """
    Args:
      num_embeddings: size of the vocabulary.
      embedding_dim: dimension of the embeddings.
      device: Device to store the parameters on.
      dtype: Data type of the parameters.
    """
    super().__init__()
    self.num_embeddings = num_embeddings
    self.embedding_dim = embedding_dim
    self.embeddings = nn.Parameter(
        torch.nn.init.trunc_normal_(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype),
            mean=0,
            std=1,
            a=-3,
            b=3,
        )
    )

  def forward(self, token_ids: torch.LongTensor) -> torch.Tensor:
    return self.embeddings[token_ids]
  
  def set_weights(self, weight: Float[Tensor, " num_embeddings embedding_dim"]):
    """Set the weights of the layer.
    
    Args:
      weight: Weights for the embedding layer.
    """
    assert weight.shape == self.embeddings.data.shape, "unexpected shape for weight: got {}, expected {}".format(weight.shape, self.embeddings.data.shape)
    self.embeddings.data = weight


class RMSNorm(nn.Module):
  """RMSNorm layer to standardize activations (like nn.RMSNorm)
  """
  def __init__(
    self,
    d_model: int,
    eps: float = 1e-5,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
  ):
    """
    Args:
      d_model: Hidden dimension of the model.
      eps: Epsilon value for numerical stability.
      device: Device to store the parameters on.
      dtype: Data type of the parameters.
    """
    super().__init__()
    self.d_model = d_model
    self.eps = eps
    self.scale = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

  def forward(self, x: Float[Tensor, " ... d_model"]) -> torch.Tensor:
    x_fp32 = x.to(torch.float32)
    rrms = torch.rsqrt(
      reduce(x_fp32.pow(2), " ... d_model -> ... 1", "mean") + self.eps
    )
    x_fp32 = x_fp32 * self.scale * rrms
    return x_fp32.type_as(x)
  
  def set_weights(self, weight: Float[Tensor, " d_model"]):
    """Set the weights of the layer.
    
    Args:
      weight: Weights for the RMSNorm layer.
    """
    assert weight.shape == self.scale.data.shape, "unexpected shape for weight: got {}, expected {}".format(weight.shape, self.scale.data.shape)
    self.scale.data = weight


def silu(in_features: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
  """SiLU activation function.
  
  Args:
    in_features: Input tensor of any shape.
  
  Returns:
    Tensor of with the same shape as `in_features` with the output of applying
    SiLU to each element.
  """
  return in_features * torch.sigmoid(in_features)


class SwiGLU(nn.Module):
  """SwiGLU activation function.
  """
  def __init__(
    self,
    d_model: int,
    d_ff: int,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
  ):
    """
    Args:
      d_model: Hidden dimension of the model and of the input of the SwiGLU.
      d_ff: Hidden of the feedforward layer. You should set d_ff approximately
        to 8/3 of d_model, while ensuring that the dimensionality of the inner
        feed-forward layer is a multiple of 64 to make good use of your
        hardware.
      device: Device to store the parameters on. dtype: Data type of the
      parameters.
    """
    super().__init__()
    self.w1 = Linear(d_model, d_ff, device, dtype)
    self.w2 = Linear(d_ff, d_model, device, dtype)
    self.w3 = Linear(d_model, d_ff, device, dtype)
  
  def forward(self, x: Float[Tensor, " ... d_model"]) -> torch.Tensor:
    return self.w2(silu(self.w1(x)) * self.w3(x))
  
  def set_weights(
    self,
    w1: Float[Tensor, " d_ff d_model"],
    w2: Float[Tensor, " d_model d_ff"],
    w3: Float[Tensor, " d_ff d_model"],
  ):
    """Set the weights of the layer.
    
    Args:
      w1: Weights for the first linear layer.
      w2: Weights for the second linear layer.
      w3: Weights for the third linear layer.
    """
    self.w1.set_weights(w1)
    self.w2.set_weights(w2)
    self.w3.set_weights(w3)


class RotaryPositionalEmbedding(nn.Module):
  """Implements Rotary Positional Embeddings (RoPE).
  """
  def __init__(
    self,
    base: float,
    d_k: int,
    max_seq_len: int,
    device: torch.device | None = None,
  ):
    """
    Args:
      base: Base value used to compute the rotation angles.
      d_k: Dimension of query and key vectors.
      max_seq_len: Maximum sequence length that will be inputted. Used to
        pre-compute and cache all the rotation angles for each position.
      device: torch.device | None = None Device to store the buffer on.
    """
    super().__init__()
    self.base = base
    self.d_k = d_k
    self.max_seq_len = max_seq_len
    self.device = device
    self.compute_cos_sin_cache()
    
  def compute_cos_sin_cache(self):
    # Generate the base angles for each position in the vector of size d_k.
    base_thetas = 1.0 / (
      self.base ** (torch.arange(0, self.d_k-1, 2).float() / self.d_k)  # d_k-1 to handle odd d_k.
    )
    # Generate the angles for each token position: multiply the angles by each
    # position.
    pos_idx = torch.arange(self.max_seq_len, device=self.device)
    pos_thetas = einsum(
      pos_idx,
      base_thetas,
      "max_seq_len, d_k_half -> max_seq_len d_k_half",
    )
    
    # Compute the cos and sin and store them. Don't compute the whole rotation
    # matrix as it is sparse and it would be memory inefficient. register_buffer
    # so that both are moved to the device along all other model parameters when
    # calling model.to(device). Do not persist these values, i.e. they won't
    # appear in the model's state_dict: the input dimensions (base, d_k,
    # max_seq_len) will have to be stored elsewhere alongside all other model's
    # hyperparameters.
    self.register_buffer("cos_cache", torch.cos(pos_thetas), persistent=False)
    self.register_buffer("sin_cache", torch.sin(pos_thetas), persistent=False)

  def forward(
    self,
    x: Float[Tensor, "... seq_len d_k"],
    token_positions: Int[Tensor, "... seq_len"] | None,
  ) -> Float[Tensor, "... seq_len d_k"]:
    
    seq_len = x.size(-2)
    
    # Get the cos and sin for each position.
    if token_positions is None:
      cos = self.cos_cache[:seq_len]  # (1, seq_len, d_k/2)
      sin = self.sin_cache[:seq_len]  # (1, seq_len, d_k/2)
    else:
      cos = self.cos_cache[token_positions]  # (..., seq_len, d_k/2)
      sin = self.sin_cache[token_positions]  # (..., seq_len, d_k/2)

    # Compute the rotation matrices for each sequence position and array pair.
    rot_matrices = rearrange(
      [cos, -sin, sin, cos],
      "(rot_row rot_col) ... seq_len d_k_half -> ... seq_len d_k_half rot_row rot_col",
      rot_row=2, rot_col=2,
    )
    
    # Reshape x to match the shape of cos_sin.
    x_pairs = rearrange(
      x,
      "... seq_len (d_k_half two) -> ... seq_len d_k_half two",
      two=2,
    )
    x_rotated = einsum(
      x_pairs,
      rot_matrices,
      "... seq_len d_k_half rot_col, ... seq_len d_k_half rot_row rot_col -> ... seq_len d_k_half rot_row",
    )

    return rearrange(x_rotated, "... seq_len d_k_half rot_row -> ... seq_len (d_k_half rot_row)")


def softmax(x: Tensor, dim: int) -> Tensor:
  """Numerically stable softmax implementation.
  
  Args:
    x: Input tensor of shape (..., d_model).
    dim: dimenstion along which to apply the softmax opeartion.
  
  Returns:
    Tensor of same shape as input with softmax applied on the last dimension.
  """
  # Since e^x can explode to inf for big x, subtract the maximum value for
  # numerical stability.
  x_exp = torch.exp(x - x.max(dim=dim, keepdim=True)[0])
  return x_exp / x_exp.sum(dim=dim, keepdim=True)
  
  
def scaled_dot_product_attention(
  Q: Float[Tensor, "... queries d_k"],
  K: Float[Tensor, "... keys d_k"],
  V: Float[Tensor, "... keys d_v"],
  mask: Bool[Tensor, "queries keys"] | None = None,
) -> Float[Tensor, "... queries d_v"]:
  """Compute the scaled dot-product attention.
  
  Args:
    Q: Query tensor.
    K: Key tensor.
    V: Value tensor.
    mask: Optional boolean mask where False values indicate positions that
      should be masked (i.e. not attended to).

  Returns:
    Output tensor after applying scaled dot-product attention.
  """
  qk = (
    einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys") /
    math.sqrt(Q.size(-1))
  ) 
  
  if mask is not None:
    qk = qk.masked_fill(~mask, float('-inf'))
  qk_softmax = softmax(qk, dim=-1)
  return einsum(
    qk_softmax,
    V,
    "... queries keys, ... keys d_v -> ... queries d_v",
  )


class MultiHeadSelfAttention(nn.Module):
  """Multi-head attention layer.
  """
  def __init__(
    self,
    d_model: int,
    num_heads: int,
    is_causal: bool = False,
    rope_max_seq_len: int | None = None,
    rope_theta: float | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
  ):
    """
    Total number of features for keys and queries (d_k) and the total number of
    features for values (d_v) are chosen so that d_k = d_v = d_model. d_model
    needs to be divisible by num_heads.

    Args:
      d_model: Hidden dimension of the model and of the input of the MHA.
      num_heads: Number of attention heads.
      is_causal: Whether to apply a causal mask to prevent attention to future
        tokens.
      rope_max_seq_len: If not None, use RoPE with the given maximum sequence
        length.
      rope_theta: If not None, use RoPE with the given base value to compute
        the rotation angles.
      device: Device to store the parameters on.
      dtype: Data type of the parameters.
    """
    super().__init__()
    assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
    
    self.d_model = d_model
    self.num_heads = num_heads
    self.is_causal = is_causal
    self.device = device
    self.dtype = dtype
    
    d_k = d_model // num_heads
    
    # Initialize RoPE if specified.
    assert (rope_max_seq_len is None) == (rope_theta is None), "both rope_max_seq_len and rope_theta should be set or both should be None"
    self.rope = None
    if rope_max_seq_len is not None:
      self.rope = RotaryPositionalEmbedding(
        rope_theta, d_k, rope_max_seq_len, device,
      )
    
    # Initialize the projection matrices.
    # Since d_k = d_v = d_model/num_heads, then Q, K, and V have the same
    # dimensions and can be stored in the same tensor. Also, for memory ordering
    # reasons, the last two dimensions should represent the (out_features,
    # in_features).
    self.q_k_v = nn.Parameter(xavier_initialized_tensor(
      [3, num_heads, d_k, d_model],
      d_model,
      d_k,
      device,
      dtype,
    ))
    self.w_o = Linear(num_heads * d_k, d_model, device, dtype)
  
  def forward(
    self,
    x: Float[Tensor, "... seq_len d_model"],
    token_positions: Int[Tensor, "... seq_len"] | None = None,
  ) -> Float[Tensor, "... seq_len d_model"]:
    
    # Multiply the input by all the 3 matrices.
    q_k_v = einsum(
      x,
      self.q_k_v,
      "... seq_len d_model, three num_head d_k d_model -> ... num_head seq_len d_k three",
    )
    # Extract the matrices to multiply q by k.
    q = q_k_v[..., 0]  # (..., num_head, seq_len, d_k)
    k = q_k_v[..., 1]  # (..., num_head, seq_len, d_k)
    v = q_k_v[..., 2]  # (..., num_head, seq_len, d_k)
    
    # Apply RoPE if specified.
    if self.rope is not None:
      q = self.rope(q, token_positions)
      k = self.rope(k, token_positions)

    mask = None
    if self.is_causal:
      # Compute the causal mask.
      seq_len = x.size(-2)
      mask = torch.tril(
        torch.ones((seq_len, seq_len), dtype=torch.bool, device=self.device)
      )
    att = scaled_dot_product_attention(q, k, v, mask)
    att = rearrange(att, "... num_head seq_len d_k -> ... seq_len (num_head d_k)")
    return self.w_o(att)
  
  def set_weights(
    self,
    q_proj_weight: Float[Tensor, " d_k_tot d_in"],
    k_proj_weight: Float[Tensor, " d_k_tot d_in"],
    v_proj_weight: Float[Tensor, " d_v_tot d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v_tot"],
  ):
    """Set the weights of the layer.
    
    Args:
      q_proj_weight: Weights for the query projection.
      k_proj_weight: Weights for the key projection.
      v_proj_weight: Weights for the value projection.
      o_proj_weight: Weights for the output projection.
    """
    d_in = q_proj_weight.size(-1)
    d_k_tot = q_proj_weight.size(-2)
    d_v_tot = v_proj_weight.size(-2)
    d_model = o_proj_weight.size(-1)

    assert d_k_tot == d_model == d_v_tot, "d_k_tot == d_model == d_v_tot is required"
    assert d_model == self.d_model, "unexpected d_model: got {}, expected {}".format(d_model, self.d_model)
    assert d_in == self.d_model, "unexpected d_in: got {}, expected {}".format(d_in, self.d_model)

    q_k_v = torch.stack(
        [
            rearrange(q_proj_weight, "(num_heads d_k) d_in -> num_heads d_k d_in", num_heads=self.num_heads),
            rearrange(k_proj_weight, "(num_heads d_k) d_in -> num_heads d_k d_in", num_heads=self.num_heads),
            rearrange(v_proj_weight, "(num_heads d_k) d_in -> num_heads d_k d_in", num_heads=self.num_heads),
        ], dim=0
    )
    assert q_k_v.shape == self.q_k_v.shape, "unexpected shape for q_k_v: got {}, expected {}".format(q_k_v.shape, self.q_k_v.shape)
    self.q_k_v.data = q_k_v

    assert o_proj_weight.shape == self.w_o.w.data.shape, "unexpected shape for o_proj_weight: got {}, expected {}".format(o_proj_weight.shape, self.w_o.w.data.shape)
    self.w_o.w.data = o_proj_weight
  

class TransformerBlock(nn.Module):
  """Transformer block with multi-head self-attention and SwiGLU feed-forward.
  """
  def __init__(
    self,
    d_model: int,
    num_heads: int,
    d_ff: int,
    rope_max_seq_len: int | None = None,
    rope_theta: float | None = None,
    rms_norm_eps: float = 1e-5,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
  ):
    """
    Args:
      d_model: Hidden dimension of the model and of the input of the Transformer
        block.
      num_heads: Number of attention heads.
      d_ff: int Dimensionality of the position-wise feed-forward inner layer.
      is_causal: Whether to apply a causal mask to prevent attention to future
        tokens.
      rope_max_seq_len: If not None, use RoPE with the given maximum sequence
        length.
      rope_theta: If not None, use RoPE with the given base value to compute
        the rotation angles.
      rms_norm_eps: Epsilon value for numerical stability in RMSNorm.
      device: Device to store the parameters on.
      dtype: Data type of the parameters.
    """
    super().__init__()
    self.rms1 = RMSNorm(d_model, rms_norm_eps, device, dtype)
    self.mha = MultiHeadSelfAttention(
      d_model,
      num_heads,
      True,
      rope_max_seq_len,
      rope_theta,
      device,
      dtype,
    )
    self.rms2 = RMSNorm(d_model, rms_norm_eps, device, dtype)
    self.ff = SwiGLU(d_model, d_ff, device, dtype)
  
  def forward(
    self,
    x: Float[Tensor, "... seq_len d_model"],
    token_positions: Int[Tensor, "... seq_len"] | None = None,
  ) -> Float[Tensor, "... seq_len d_model"]:
    x = x + self.mha(self.rms1(x), token_positions)
    x = x + self.ff(self.rms2(x))
    return x

  def set_weights(
    self,
    rms1_scale: Float[Tensor, " d_model"],
    attn_q_proj_weight: Float[Tensor, " d_k_tot d_in"],
    attn_k_proj_weight: Float[Tensor, " d_k_tot d_in"],
    attn_v_proj_weight: Float[Tensor, " d_v_tot d_in"],
    attn_o_proj_weight: Float[Tensor, " d_model d_v_tot"],
    rms2_scale: Float[Tensor, " d_model"],
    ffn_w1: Float[Tensor, " d_ff d_model"],
    ffn_w2: Float[Tensor, " d_model d_ff"],
    ffn_w3: Float[Tensor, " d_ff d_model"],
  ):
    """Set the weights of the layer.
    
    Args:
      rms1_scale: Weights for the first RMSNorm layer.
      attn_q_proj_weight: Weights for the query projection.
      attn_k_proj_weight: Weights for the key projection.
      attn_v_proj_weight: Weights for the value projection.
      attn_o_proj_weight: Weights for the output projection.
      rms2_scale: Weights for the second RMSNorm layer.
      ffn_w1: Weights for the first linear layer.
      ffn_w2: Weights for the second linear layer.
      ffn_w3: Weights for the third linear layer.
    """
    self.rms1.set_weights(rms1_scale)
    self.mha.set_weights(
        attn_q_proj_weight,
        attn_k_proj_weight,
        attn_v_proj_weight,
        attn_o_proj_weight,
    )
    self.rms2.set_weights(rms2_scale)
    self.ff.set_weights(ffn_w1, ffn_w2, ffn_w3)