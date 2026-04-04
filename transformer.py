import torch

from torch import Tensor
from causal_attention import MultiHeadAttention
from gelu import FeedForward
from gpt_model import GptModelConfig
from norm import LinearNorm

class Transformer(torch.nn.Module):
  def __init__(self, cfg: GptModelConfig):
    super().__init__()

    self.attention = MultiHeadAttention(
      d_in=cfg.embedding_dim,
      d_out=cfg.embedding_dim,
      context_length=cfg.context_length,
      num_heads=cfg.num_heads,
      drop_rate=cfg.attention_drop_rate,
      bias=cfg.qkv_bias
    )

    self.ff = FeedForward(cfg)
    self.norm1 = LinearNorm(cfg.embedding_dim)
    self.norm2 = LinearNorm(cfg.embedding_dim)
    self.drop_shortcut = torch.nn.Dropout(cfg.transformer_drop_rate)

  def __call__(self, x: Tensor) -> Tensor:
    return super().__call__(x)
  
  
  def forward(self, x: Tensor):
    shortcut = x
    x = self.norm1(x)
    x = self.attention(x)
    x = self.drop_shortcut(x)
    x = x + shortcut

    shortcut = x
    x = self.norm2(x)
    x = self.ff(x)
    x = self.drop_shortcut(x)
    x = x + shortcut

    return x
