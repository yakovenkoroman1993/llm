import torch

from cfg import GPT_CONFIG_124M
from components.gpt_model import GptModel

NUM_CLASSES = 2 # 0 vs 1 or yes/no

class ClassificationGptModel(GptModel):
  def __init__(self, cfg):
    super().__init__(cfg)

    # FINE TUNING START 
    # Замораживаем все слои как необучаемые
    for param in self.parameters():
      param.requires_grad = False
    for param in self.transformer_blocks[-1].parameters():
      param.requires_grad = True
    for param in self.final_norm.parameters():
      param.requires_grad = True

    # torch.manual_seed(123)
    self.out_head = torch.nn.Linear(
      in_features=GPT_CONFIG_124M.embedding_dim,
      out_features=NUM_CLASSES
    )
    # FINE TUNING END
    
