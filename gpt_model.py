import torch

from typing import Optional
from torch import Tensor
from cfg import GptModelConfig
from norm import LinearNorm
from transformer import Transformer


class GptModel(torch.nn.Module):
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
        Transformer(cfg)
        for _ in range(cfg.num_layers)
      ]
    )

    self.final_norm = LinearNorm(cfg.embedding_dim)

    self.out_head = torch.nn.Linear(
      in_features=cfg.embedding_dim,
      out_features=cfg.vocab_size,
      bias=cfg.qkv_bias # 
      # bias=False
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
    x = self.final_norm(x)

    logits: Tensor = self.out_head(x)

    return logits
  
  # Генерация токенов на базе методов масштабирования температур и top-k
  def generate(
    self,
    idx: Tensor,
    max_new_tokens: int,
    context_size: int,
    temperature = 0.0,
    top_k: Optional[int] = None,
    eos_id = None
  ):
    for _ in range(max_new_tokens):
      idx_cond = idx[:, -context_size:]

      with torch.no_grad():
        logits = self(idx_cond)
      
      logits = logits[:, -1, :]

      if top_k is not None:
        top_logits, _ = torch.topk(logits, top_k)
        min_val = top_logits[:, -1]
        logits = torch.where(
          condition=logits < min_val,
          input=torch.tensor(float('-inf')).to(logits.device),
          other=logits,
        )
      if temperature > 0.0:
        logits = logits / temperature
        probs = torch.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
      else:
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)
      
      if idx_next == eos_id:
        break

      idx = torch.cat((idx, idx_next), dim=1)
    
    return idx
    

  # used greedy decoding
  def generate_greedy(
    self,
    idx: Tensor,
    max_new_tokens: int,
    context_size: int
  ):
    for _ in range(max_new_tokens):
      idx_cond = idx[:, -context_size:]

      with torch.no_grad():
        logits = self(idx_cond)
      
      logits = logits[:, -1, :]
      probas = torch.softmax(logits, dim=-1)
      
      idx_next = torch.argmax(probas, dim=-1, keepdim=True)
      idx = torch.concat((idx, idx_next), dim=-1)
    
    return idx

