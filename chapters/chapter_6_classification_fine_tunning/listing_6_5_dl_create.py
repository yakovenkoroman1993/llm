import sys

sys.path.insert(0, "/Users/roman/WebstormProjects/llm")

import tiktoken
import torch
from torch.utils.data import DataLoader
from chapters.chapter_6_classification_fine_tunning.listing_6_4_spam_dataset_class import SpamDataset

tokenizer = tiktoken.get_encoding("gpt2")
num_workers = 0
batch_size = 8
torch.manual_seed(123)

train_dataset = SpamDataset(
  csv_file="train_data/train.csv",
  max_length=None,
  tokenizer=tokenizer
)

print(train_dataset.max_length)

val_dataset = SpamDataset(
  csv_file="train_data/validation.csv",
  max_length=train_dataset.max_length,
  tokenizer=tokenizer
)
test_dataset = SpamDataset(
  csv_file="train_data/test.csv",
  max_length=train_dataset.max_length,
  tokenizer=tokenizer
)

train_loader = DataLoader(
  dataset=train_dataset,
  batch_size=batch_size,
  shuffle=True,
  num_workers=num_workers,
  drop_last=True,
)
val_loader = DataLoader(
  dataset=val_dataset,
  batch_size=batch_size,
  num_workers=num_workers,
  drop_last=False,
)
test_loader = DataLoader(
  dataset=test_dataset,
  batch_size=batch_size,
  num_workers=num_workers,
  drop_last=False,
)

for input_batch, target_batch in train_loader:
  pass
print("Input batch dimensions:", input_batch.shape)
print("Label batch dimensions", target_batch.shape)

print(f"{len(train_loader)} training batches")
print(f"{len(val_loader)} validation batches")
print(f"{len(test_loader)} test batches")

# input_batch, target_batch = next(iter(train_loader))
# print("Input batch dimensions:", input_batch.shape)
# print("Label batch dimensions", target_batch.shape)