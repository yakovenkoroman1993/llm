  
from gpt_model import GptModel


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
