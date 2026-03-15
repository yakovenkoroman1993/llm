from importlib.metadata import version
import dl
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
output_dim = 256    # vector dimension

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
print(token_embeddings)
print("SHAPE", token_embeddings.shape)
print(position_embeddings)
print("SHAPE", position_embeddings.shape)
input_embeddings = token_embeddings + position_embeddings
print(input_embeddings)
print("SHAPE", input_embeddings.shape)
 
print("End")
