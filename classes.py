
from dataclasses import dataclass
from typing import Any, Mapping, Optional
from torch.optim.optimizer import StateDict as OptimStateDict

@dataclass
class GptModelConfig:
  vocab_size: int                       # относится к словарю из 50 257 слов, используемому токенизатором BPE
  context_length: int                   # максимальное количество входных токенов
  embedding_dim: int                    # размерность вложения, преобразующего каждый токен в вектор размерностью 768 элементов
  num_heads: int                        # количество целей в механизме многоцелевого внимания
  num_layers: int                       # количество блоков трансформера в модели
  qkv_bias: bool                        # определяет, следует ли добавлять вектор смещения в слои Linear многоцелевого внимания для вычисления запроса, ключа и значения.
  drop_rate: float                      # интенсивность механизма отсева для предотвращения переобучения 
  transformer_drop_rate: float          # Отсев в трансформере
  attention_drop_rate: float            # Отсев в механизме многоцелевого внимания MultiHeadAttention

@dataclass
class GptModelProgress:
  model_state_dict: Mapping[str, Any]
  optim_state_dict: Optional[OptimStateDict]


@dataclass
class ImportModelConfig:
  emb_dim: int
  n_layers: int
  n_heads: int
