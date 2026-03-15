from typing import Iterator
import tiktoken
import torch
from torch import Tensor
from torch.utils.data import Dataset

# Типизированный torch DataLoader
class DataLoader(torch.utils.data.DataLoader[tuple[Tensor, Tensor]]):
  def __iter__(self) -> Iterator[tuple[Tensor, Tensor]]:
    return super().__iter__()

# Токенизация и Реализация плавающего окна (входные данные - цель)
class SlidingWindow(Dataset[tuple[Tensor, Tensor]]):
  def __init__(
      self, 
      txt: str, 
      tokenizer: tiktoken.Encoding, 
      max_length: int, 
      stride: int
  ):
    self.input_ids = []
    self.target_ids = []

    token_ids = tokenizer.encode(txt)

    for i in range(0, len(token_ids) - max_length, stride):
      input_chunk = token_ids[i: i + max_length]
      self.input_ids.append(torch.tensor(input_chunk))

      target_chunk = token_ids[i + 1: i + max_length + 1]
      self.target_ids.append(torch.tensor(target_chunk))
  
  def __len__(self):
    return len(self.input_ids)

  def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
    return self.input_ids[index], self.target_ids[index]
  
  
def create (
  txt: str,
  batch_size = 4,
  max_length = 256,
  stride = 128,
  shuffle = False,
  drop_last = True,
):
  tokenizer = tiktoken.get_encoding("gpt2")

  dataset = SlidingWindow(
    txt=txt,
    tokenizer=tokenizer,
    max_length=max_length,
    stride=stride,
  )
  return DataLoader(
    dataset=dataset,
    batch_size=batch_size,
    shuffle=shuffle,
    drop_last=drop_last,
    num_workers=0,
  )
