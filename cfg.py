
from typing import Literal

from classes import GptModelConfig, ImportModelConfig
from aliases import LlmSize

GPT_CONFIG_PROGRESS = GptModelConfig(
  embedding_dim=768,
  num_layers=12,
  num_heads=12,
  vocab_size=50257,
  context_length=256, # чтобы уменьшить нагрузку при обучении локально [ML]
  drop_rate=0.1,
  transformer_drop_rate=0.1,
  attention_drop_rate=0.1,
  qkv_bias=False # [ML]
)

# "emb_dim": 768, 
# "n_layers": 12, 
# "n_heads": 12
GPT_CONFIG_124M = GptModelConfig(
  embedding_dim=768,
  num_layers=12,
  num_heads=12,
  vocab_size=50257,
  context_length=1024, # Valid loss = nan, perplexity = nan
  # context_length=256, # чтобы уменьшить нагрузку при обучении локально [ML]
  drop_rate=0.1,
  transformer_drop_rate=0.1,
  attention_drop_rate=0.1,
  # qkv_bias=False # [ML]
  
  # Векторы смещения больше не применяются в LLM,
  # так как не улучшают производительность моделирования и, следовательно,
  # не нужны. Однако мы работаем с предварительно обученными весами, поэтому в целях единообразия нужно согласовать настройки 
  # и включить векторы смещения
  qkv_bias=True
)

# "emb_dim": 1024, 
# "n_layers": 24, 
# "n_heads": 16
GPT_CONFIG_355M = GptModelConfig(
  embedding_dim=1024,
  num_layers=24,
  num_heads=16,
  vocab_size=50257,
  context_length=1024,
  drop_rate=0.1,
  transformer_drop_rate=0.1,
  attention_drop_rate=0.1,
  qkv_bias=True
)

# "emb_dim": 1280, 
# "n_layers": 36, 
# "n_heads": 20
GPT_CONFIG_774M = GptModelConfig(
  embedding_dim=1280,
  num_layers=36,
  num_heads=20,
  vocab_size=50257,
  context_length=1024,
  drop_rate=0.1,
  transformer_drop_rate=0.1,
  attention_drop_rate=0.1,
  qkv_bias=True
)

# "emb_dim": 1600, 
# "n_layers": 48, 
# "n_heads": 25
GPT_CONFIG_1558M = GptModelConfig(
  embedding_dim=1600,
  num_layers=48,
  num_heads=25,
  vocab_size=50257,
  context_length=1024,
  drop_rate=0.1,
  transformer_drop_rate=0.1,
  attention_drop_rate=0.1,
  qkv_bias=True
)

GPT_MODEL_CONFIGS: dict[LlmSize, GptModelConfig]  = {
  "progress": GPT_CONFIG_PROGRESS,
  "124M": GPT_CONFIG_124M,
  "355M": GPT_CONFIG_355M,
  "774M": GPT_CONFIG_774M,
  "1558M": GPT_CONFIG_1558M,
}