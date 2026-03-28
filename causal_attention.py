import torch
from torch import Tensor

class CausalAttention(torch.nn.Module):
  def __init__(
    self,
    d_in: int,
    d_out: int,
    context_length: int,
    drop_rate: float,
    bias=False
  ):
    super().__init__()
    self.d_out = d_out
    self.W_query = torch.nn.Linear(d_in, d_out, bias)
    self.W_key = torch.nn.Linear(d_in, d_out, bias)
    self.W_value = torch.nn.Linear(d_in, d_out, bias)
    self.dropout = torch.nn.Dropout(drop_rate)
    self.register_buffer(
      "mask",
      torch.triu(torch.ones(context_length, context_length), diagonal=1)
    )
  
  def __call__(self, x: Tensor) -> Tensor:
    return super().__call__(x)

  def forward(self, x: Tensor):
    batch_size, num_tokens, d_in = x.shape

    queries: Tensor = self.W_query(x)
    keys: Tensor = self.W_key(x)
    values: Tensor = self.W_value(x)

    attention_scores = queries @ keys.transpose(1, 2)
    attention_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
    attention_weights = torch.softmax(attention_scores / keys.shape[-1]**0.5, dim=-1)
    attention_weights: Tensor = self.dropout(attention_weights)

    return attention_weights @ values


class MultiHeadAttention_v1(torch.nn.Module):
  def __init__(
    self, 
    d_in: int, 
    d_out: int,
    context_length: int,
    drop_rate: float,
    num_heads: int,
    bias = False
  ):
    super().__init__()
    
    self.heads = torch.nn.ModuleList(
      [
        CausalAttention(
          d_in,
          d_out,
          context_length,
          drop_rate,
          bias
        )
        for _ in range(num_heads)
      ]
    )
  
  def forward(self, batch: Tensor):
    return torch.cat(
      [
        head(batch) for head in self.heads
      ],
      dim=-1
    )


class MultiHeadAttention(torch.nn.Module):
  def __init__(
    self, 
    d_in: int, 
    d_out: int,
    context_length: int,
    drop_rate: float,
    num_heads: int,
    bias = False
  ):
    super().__init__()
    
    assert(d_out % num_heads == 0, "d_out must be divisble by num_heads")

    self.d_out = d_out
    self.num_heads = num_heads
    self.head_dim = d_out // num_heads

    self.W_query = torch.nn.Linear(d_in, d_out, bias)
    self.W_key = torch.nn.Linear(d_in, d_out, bias)
    self.W_value = torch.nn.Linear(d_in, d_out, bias)

    self.out = torch.nn.Linear(d_out, d_out) # обучаемый слой для связы голов на выходе

    self.dropout = torch.nn.Dropout(drop_rate)
    self.register_buffer(
      "mask",
      torch.triu(torch.ones(context_length, context_length), diagonal=1)
    )

  
  def forward(self, x: Tensor):
    b, num_tokens, d_in = x.shape

    # Линейные проекции — три матрицы весов применяются к входу
    keys: Tensor = self.W_key(x)
    queries: Tensor = self.W_query(x)
    values: Tensor = self.W_value(x)

    # Разбивка на num_heads голов — reshape + transpose, без копирования данных:
    #    b, num_tokens, d_in 
    # -> b, num_tokens, num_heads, head_dim 
    # -> b, num_heads, num_tokens, head_dim
    # reshape вместо view чтобы избежать Runtime error: RuntimeError: non-contiguous
    keys = keys \
      .reshape(b, num_tokens, self.num_heads, self.head_dim) \
      .transpose(1, 2) 
    
    queries = queries \
      .reshape(b, num_tokens, self.num_heads, self.head_dim) \
      .transpose(1, 2)
    
    values = values \
      .reshape(b, num_tokens, self.num_heads, self.head_dim) \
      .transpose(1, 2)
  
    # Оценки внимания 
    attention_scores = queries @ keys.transpose(2, 3)
    
    # Каузальная маска
    causal_mask = self.mask.bool()[:num_tokens, :num_tokens]
    attention_scores.masked_fill_(causal_mask, -torch.inf)

    # Масштабирование и softmax
    attention_weights = torch.softmax(attention_scores / keys.shape[-1]**0.5, dim=-1)

    attention_weights: Tensor = self.dropout(attention_weights)

    # Взвешенная сумма значений
    #  -> b, num_heads, num_tokens, head_dim
    #  -> b, num_tokens, num_heads, head_dim 
    #  -> конкатенация голов (приведение к начальной форме входных данных)
    context = (attention_weights @ values) \
      .transpose(1, 2) \
      .reshape(b, num_tokens, self.d_out)
    
    return self.out(context) # каждый элемент теперь зависит от всех голов

    
