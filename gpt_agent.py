from typing import Optional

import torch

from tiktoken import Encoding
from torch import Tensor, dtype
from encoder import Encoder
from gpt_model import GptModel

class GptModelAgent():
  def __init__(
    self,
    model: GptModel,
    tokenizer: Encoding,
    device: dtype,
  ):
    self.model = model
    self.encoder = Encoder(tokenizer)
    self.device = device

  def send_message(
    self,
    message: str,
    device: Optional[dtype] = None,
    temperature = 1.4,
    top_k = 25,

  ) -> str:
    self.model.eval() # отключение обучения, отсева Dropout

    context_size = self.model.position_embedding.weight.shape[0]

    encoded = self.encoder \
      .text_to_token_ids(message) \
      .to(device or self.device)

    with torch.no_grad():
      token_ids = self.model.generate(
        idx=encoded,
        max_new_tokens=50,
        context_size=context_size,
        temperature=temperature,
        top_k=top_k,
      )

      decoded_text = self.encoder.token_ids_to_text(token_ids)      

    self.model.train()

    return decoded_text

  
