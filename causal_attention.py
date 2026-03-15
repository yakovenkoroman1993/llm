import torch
from torch import Tensor

class CausalAttention(torch.nn.Module):
  def __init__(
    self,
    d_in: int,
    d_out: int,
    context_length: int,
    dropout: float,
    bias=False
  ):
    super().__init__()
    self.d_out = d_out
    self.W_query = torch.nn.Linear(d_in, d_out, bias)
    self.W_key = torch.nn.Linear(d_in, d_out, bias)
    self.W_value = torch.nn.Linear(d_in, d_out, bias)
    self.dropout = torch.nn.Dropout(dropout)
    self.register_buffer(
      "mask",
      torch.triu(torch.ones(context_length, context_length), diagonal=1)
    )
  
  def __call__(self, batch: Tensor) -> Tensor:
    return super().__call__(batch)

  def forward(self, batch: Tensor):
    batch_size, num_tokens, d_in = batch.shape

    queries: Tensor = self.W_query(batch)
    keys: Tensor = self.W_key(batch)
    values: Tensor = self.W_value(batch)

    attention_scores = queries @ keys.transpose(1, 2)
    attention_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
    attention_weights = torch.softmax(attention_scores / keys.shape[-1]**0.5, dim=-1)
    attention_weights: Tensor = self.dropout(attention_weights)

    return attention_weights @ values
