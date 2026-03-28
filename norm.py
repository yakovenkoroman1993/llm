import torch
from torch import Tensor

# Это слой предобработки данных, а не функция активации. 
# Его задача — сделать значения «удобными» для нейросети, чтобы градиенты не взрывались и не затухали.
class LinearNorm(torch.nn.Module):
  def __init__(self, embedding_dim: int):
    super().__init__()

    self.eps = 1e-5

    self.scale = torch.nn.Parameter(torch.ones(embedding_dim))
    self.shift = torch.nn.Parameter(torch.zeros(embedding_dim))

  def __call__(self, x: Tensor) -> Tensor:
    return super().__call__(x)
  
  def forward(self, x: Tensor):
    mean = x.mean(dim=-1, keepdim=True)
    variance = x.var(dim=-1, keepdim=True, unbiased=False) # поправка Бесселя выключенав. Делится на n вместо n-1
    norm_x = (x - mean) / torch.sqrt(variance + self.eps) 
    
    # Теперь Среднее = 0 и Дисперсия = 1, а.значит градиенты (производные) не затухнут т.е. не будут стремится к 0
    return self.scale * norm_x + self.shift