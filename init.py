from importlib.metadata import version
import dl
from self_attention import SelfAttention_v1, SelfAttention_v2
from causal_attention import CausalAttention, MutliHeadAttention, MutliHeadAttention_v2
import torch
from torch import Tensor
from typing import Iterator

print("tiktoken version:", version("tiktoken"))

with open("the-verdict.txt", "r", encoding="utf-8") as f:
  raw_text = f.read()

# 1.0 Создаем итератор для пакетов с токенами: 1 пакет = 8 последовательностей по 4 токена
max_length = 4
data_loader = dl.create(
  raw_text,
  batch_size=8,
  max_length=max_length,
  stride=max_length,
)

data_iter = iter(data_loader)
inputs, targets = next(data_iter)


# 2.0 Создаем нейронный слой Вложения токенов (веса случайные)
vocab_size = 50257  # токенизатора BPE vocab size
# output_dim = 256    # vector dimension
output_dim = 1024    # как у GPT-2

torch.manual_seed(123) # постоянная детерминированность. Удобно для отладки

token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

# 2.1 Получаем Вложения токенов: каждый токен представляется как вектор из слоя Вложения, взятый по индексу
token_embeddings: Tensor = token_embedding_layer(inputs)

# 2.2 Создаем нейронный слой Позиционные Вложения
context_length = max_length
positions: Tensor = torch.arange(context_length)

# 2.3 Получаем Позиционные Вложения
position_embedding_layer = torch.nn.Embedding(context_length, output_dim)
position_embeddings: Tensor = position_embedding_layer(torch.arange(context_length))

# 3.3 Получим Входные вложения из суммы: Вложения токенов и Позиционные Вложения
input_embeddings = token_embeddings + position_embeddings
 
# Глава 3. Самовнимание или получение контекстных векторов для каждого токена
# 4.0

inputs = input_embeddings[0]

# inputs = torch.tensor(
#   [
#     [0.43, 0.15, 0.89], # Your (x1)
#     [0.55, 0.87, 0.66], # journey (x2)
#     [0.57, 0.85, 0.64], # starts (x3)
#     [0.22, 0.58, 0.33], # with (x4)
#     [0.77, 0.25, 0.10], # one (x5)
#     [0.05, 0.80, 0.55], # step (x6)
#   ]
# )

torch.manual_seed(789)
sa_v1 = SelfAttention_v1(
  d_in=inputs.shape[1], 
  d_out=2
)

torch.manual_seed(789)
sa_v2 = SelfAttention_v2(
  d_in=inputs.shape[1],
  d_out=2
)

batch = torch.stack((inputs, inputs), dim=0)

ca = CausalAttention(
  d_in=batch.shape[2],
  d_out=2, 
  context_length=batch.shape[1],
  dropout=0.0
)

context_vecs = ca(batch)

# Многоцелевое внимание

torch.manual_seed(123)
mha = MutliHeadAttention(
  d_in=batch.shape[2],
  d_out=2,
  context_length=batch.shape[1],
  dropout=0,
  num_heads=2
)

context_vecs = mha(batch)

torch.manual_seed(123)
mha = MutliHeadAttention_v2(
  d_in=batch.shape[2], #3
  d_out=768, # GPT-2
  context_length=batch.shape[1],
  dropout=0.0,
  num_heads=12 # GPT-2
)

context_vecs = mha(batch)
print(context_vecs)