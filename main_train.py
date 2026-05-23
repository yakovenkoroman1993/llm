import os
import tiktoken
import argparse
import torch
import components.dl as dl

from cfg import GPT_CONFIG_PROGRESS
from components.evaluator import ModelEvaluator
from components.gpt_model import GptModel
from components.gpt_agent import GptModelAgent
from components.ml import MachineLearning
from classes import GptModelProgress
from dataclasses import dataclass

DEFAULT_LLM_SOURCE = "train_data/the-verdict.txt"
DEFAULT_LLM_FILE = "progress.pth"

def run(
  llm_file = DEFAULT_LLM_FILE,
  llm_source = DEFAULT_LLM_SOURCE
):
  with open(llm_source, "r", encoding="utf-8") as f:
    raw_text = f.read()

  tokenizer = tiktoken.get_encoding("gpt2")

  train_ratio = 0.9
  split_position = int(train_ratio * len(raw_text))
  train_data = raw_text[:split_position]
  valid_data = raw_text[split_position:]

  # Создаем итератор для пакетов с токенами: 1 пакет = 2 последовательности по 256 токена
  train_loader = dl.DataLoader(
    dataset=dl.SlidingWindow(
      txt=train_data,
      tokenizer=tokenizer,
      max_length=GPT_CONFIG_PROGRESS.context_length,
      stride=GPT_CONFIG_PROGRESS.context_length,
    ),
    batch_size=2,
    shuffle=True,
    drop_last=True,
    num_workers=0,
  )

  valid_loader = dl.DataLoader(
    dataset=dl.SlidingWindow(
      txt=valid_data,
      tokenizer=tokenizer,
      max_length=GPT_CONFIG_PROGRESS.context_length,
      stride=GPT_CONFIG_PROGRESS.context_length,
    ),
    batch_size=2,
    shuffle=False,
    drop_last=False,
    num_workers=0,
  )

  # torch.manual_seed(123)

  def handle_epoch(device: torch.dtype):
    answer = agent \
      .send_message("Every effort moves you", device) \
      .replace('\n', ' ')

    print(
      f"\n***\n"
      f"Gpt Agent Response: {answer}"
      f"\n***\n" 
    ),

  device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
  # device = "cpu"

  if os.path.exists(llm_file):
    torch.serialization.add_safe_globals([GptModelProgress])
    progress: GptModelProgress = torch.load(
      llm_file,
      map_location=device
    )
  else:
    progress = None  

  model = GptModel(GPT_CONFIG_PROGRESS)

  if progress is not None:
    model.load_state_dict(progress.model_state_dict)

  model.to(device)

  optimAdamW = torch.optim.AdamW(
    params=model.parameters(),
    lr=0.0004,
    weight_decay=0.1,
  )

  if progress is not None:
    if progress.optim_state_dict is not None:
      optimAdamW.load_state_dict(progress.optim_state_dict)

  ml = MachineLearning(
    model=model,
    device=device,
    optimizer=optimAdamW,
    train_loader=train_loader,
    valid_loader=valid_loader,
    evaluator=ModelEvaluator(
      model=model,
      train_loader=train_loader,
      valid_loader=valid_loader,
      device=device
    )
  )

  agent = GptModelAgent(
    model=model,
    tokenizer=tokenizer,
    device=device,
  )

  train_losses, valid_losses, tokens_seen = ml.train_model(
    num_epochs=10,
    eval_freq=5,
    eval_num_batches=5,
    on_epoch=lambda device: handle_epoch(device),
    on_batch=lambda *args, **kwargs: \
      ModelEvaluator.show_losses(*args, **kwargs)
  )

  torch.save(
    GptModelProgress(
      model_state_dict=model.state_dict(),
      optim_state_dict=optimAdamW.state_dict(),
    ),
    llm_file
  )

  ModelEvaluator.plot_losses(
    train_losses=train_losses,
    valid_losses=valid_losses,
    amount_seen=tokens_seen
  )

if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  parser.add_argument("--llm-file", default=DEFAULT_LLM_FILE)
  parser.add_argument("--llm-source", default=DEFAULT_LLM_SOURCE)

  @dataclass
  class ArgsNamespace:
    llm_file: str
    llm_source: str

  args = parser.parse_args(namespace=ArgsNamespace)

  if not os.path.exists(args.llm_source):
    raise ValueError(f"llm файл источника не найден: {args.llm_source}")

  run(
    llm_file=args.llm_file,
    llm_source=args.llm_source
  )