import torch

from typing import Optional
from torch import dtype
from components.dl import DEFAULT_PAD_TOKEN_ID
from components.encoder import Encoder
from components.gpt_agent import GptModelAgent

class ClassificationGptModelAgent(GptModelAgent):
  def send_message(
    self,
    message: str,
    device: Optional[dtype] = None,
    max_length=768
  ) -> str:
    self.model.eval() # отключение обучения, отсева Dropout

    encoded = self.encoder \
      .text_to_token_ids(message) \
      # .to(device or self.device)
      
    encoded = encoded.squeeze().tolist()

    supported_context_length: int = self.model.position_embedding.weight.shape[1]

    # Усекает слишком длинные последовательности
    encoded = encoded[:min(
      max_length, supported_context_length
    )]

    encoded += [DEFAULT_PAD_TOKEN_ID] * (max_length - len(encoded))

    input_tensor = torch.tensor(
      encoded, 
      device=device
    ).unsqueeze(0)
    
    with torch.no_grad():
      logits = self.model(input_tensor)[:, -1, :] # Логиты последнего выходного токена
    
    predicted_label = torch.argmax(logits, dim=-1).item()

    return "spam" if predicted_label == 1 else "not spam"
    

  
