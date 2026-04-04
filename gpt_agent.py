import torch

from tiktoken import Encoding
from torch import Tensor, dtype
from encoder import Encoder
from gpt import GptModel

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
    device: dtype,
  ):
    self.model.eval() # отключение обучения, отсева Dropout

    context_size = self.model.position_embedding.weight.shape[0]

    encoded = self.encoder \
      .text_to_token_ids(message) \
      .to(device or self.device)

    with torch.no_grad():
      token_ids = self.__gen_text_simple(
        idx=encoded,
        max_new_tokens=50,
        context_size=context_size,
      )

      decoded_text = self.encoder \
        .token_ids_to_text(token_ids) \
        .replace("\n", " ")
      
      print(
        f"\n***\n"
        f"Gpt Agent Response: {decoded_text}"
        f"\n***\n" 
      )

    self.model.train()

  def __gen_text_simple(
    self,
    idx: Tensor,
    max_new_tokens: int,
    context_size: int
  ):
    for _ in range(max_new_tokens):
      idx_cond = idx[:, -context_size:]

      with torch.no_grad():
        logits = self.model(idx_cond)
      
      logits = logits[:, -1, :]
      probas = torch.softmax(logits, dim=-1)
      idx_next = torch.argmax(probas, dim=-1, keepdim=True)
      idx = torch.concat((idx, idx_next), dim=-1)
    
    return idx
