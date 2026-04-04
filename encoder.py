from tiktoken import Encoding
from torch import Tensor
import torch

class Encoder():
  def __init__(self, tokenizer: Encoding):
    self.tokenizer = tokenizer

  def text_to_token_ids(self, text: str):
    encoded = self.tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    return torch.tensor(encoded).unsqueeze(0)

  def token_ids_to_text(self, token_ids: Tensor):
    flat = token_ids.squeeze(0)
    return self.tokenizer.decode(flat.tolist())
