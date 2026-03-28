import torch
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

  
def print_num_of_paramenters(model: GptModel):
  fake_total_params = sum(p.numel() for p in model.parameters())
  print("fake_total_params", fake_total_params)

  print("Token embedding layer shape:", model.token_embedding.weight.shape)
  print("Output layer shape:", model.out_head.weight.shape)

  honest_total_params = fake_total_params - sum(
    p.numel() for p in model.out_head.parameters()
  )

  print("honest_total_params", honest_total_params)

  numel_transformer_blocks = sum(
    p.numel() 
    for p in model.transformer_blocks.parameters()
  )

  numel_transformer_blocks2 = sum(
    sum(
      p.numel() 
      for b in model.transformer_blocks
      for p in getattr(b, name).parameters()
    )
    for name in ("attention", "ff", "norm1", "norm2", "drop_shortcut")
  )

  print("numel_transformer_blocks", numel_transformer_blocks)
  print("numel_transformer_blocks2", numel_transformer_blocks2)
