import torch
from torch import Tensor
from dataclasses import dataclass

@dataclass
class GptModelConfig:
  vocab_size: int       # относится к словарю из 50 257 слов, используемому токенизатором BPE
  context_length: int   # максимальное количество входных токенов
  embedding_dim: int    # размерность вложения, преобразующего каждый токен в вектор размерностью 768 элементов
  num_heads: int        # количество целей в механизме многоцелевого внимания
  num_layers: int       # количество блоков трансформера в модели
  drop_rate: float      # интенсивность механизма отсева для предотвращения переобучения 
  qkv_bias: bool        # определяет, следует ли добавлять вектор смещения в слои Linear многоцелевого внимания для вычисления запроса, ключа и значения.


class DummyGptModel(torch.nn.Module):
  def __init__(self, cfg: GptModelConfig):
    super().__init__()

    self.token_embedding = torch.nn.Embedding(
      num_embeddings=cfg.vocab_size,
      embedding_dim=cfg.embedding_dim,
    )
    
    self.position_embedding = torch.nn.Embedding(
      num_embeddings=cfg.context_length,
      embedding_dim=cfg.embedding_dim,
    )

    self.drop_embedding = torch.nn.Dropout(cfg.drop_rate)

    self.transformer_blocks = torch.nn.Sequential(
      *[
        DummyTransformerBlock(cfg)
        for _ in range(cfg.num_layers)
      ]
    )

    self.final_norm = DummyLayerNorm(cfg.embedding_dim)

    self.out_head = torch.nn.Linear(
      in_features=cfg.embedding_dim,
      out_features=cfg.vocab_size,
      bias=cfg.qkv_bias # False
    )

  def __call__(self, x: Tensor) -> Tensor:
    return super().__call__(x)

  def forward(self, in_idx: Tensor):
    batch_size, seq_len = in_idx.shape

    token_embeddings: Tensor = self.token_embedding(in_idx)
    position_embeddings: Tensor = self.position_embedding(
      torch.arange(seq_len, device=in_idx.device)
    )

    x = token_embeddings + position_embeddings
    x = self.drop_embedding(x)
    x = self.transformer_blocks(x)
    logits: Tensor = self.out_head(x)

    return logits


class DummyTransformerBlock(torch.nn.Module):
  def __init__(self, cfg: GptModelConfig):
    super().__init__()

  def forward(self, x: Tensor):
    return x
  

class DummyLayerNorm(torch.nn.Module):
  def __init__(
    self, 
    normalized_shape,
    eps=1e-5
  ):
    super().__init__()

  def forward(self, x: Tensor):
    return x
  