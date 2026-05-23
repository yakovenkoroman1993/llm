import os
import time
import tiktoken
import argparse
import torch
import components.dl as dl

from cfg import GPT_CONFIG_124M
from components.evaluator_classification import ClassificationModelEvaluator
from classes import GptModelProgress
from dataclasses import dataclass, replace

from components.gpt_model_classification import ClassificationGptModel
from components.ml_classification import ClassificationMachineLearning

DEFAULT_LLM_TRAIN_SOURCE = "train_data/train.csv"
DEFAULT_LLM_VALID_SOURCE = "train_data/validation.csv"
DEFAULT_LLM_FILE = "gpt2_124m_spam.pth"

BATCH_SIZE=8
NUM_WORKERS=0

def run(
  llm_file: str,
  llm_train_source = DEFAULT_LLM_TRAIN_SOURCE,
  llm_valid_source = DEFAULT_LLM_VALID_SOURCE,
):
  tokenizer = tiktoken.get_encoding("gpt2")

  # Создаем итератор для пакетов с токенами: 1 пакет = 8 последовательностей по 120 токена (120 = train_dataset.max_length)
  train_dataset = dl.SpamDataset(
    csv_file=llm_train_source,
    max_length=None,
    tokenizer=tokenizer
  )
  train_loader = dl.DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    shuffle=True,
    drop_last=True,
  )

  valid_loader = dl.DataLoader(
    dataset=dl.SpamDataset(
      csv_file=llm_valid_source,
      max_length=train_dataset.max_length,
      tokenizer=tokenizer
    ),
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    drop_last=False,
  )

  def handle_epoch(
    train_accuracy: float, 
    valid_accuracy: float
  ):
    print(f"Training accuracy: {train_accuracy*100:.2f}% | ", end="")
    print(f"Validation accuracy: {valid_accuracy*100:.2f}%")

  device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
  # device = "cpu"

  if os.path.exists(llm_file):
    print(f"LLM \"{llm_file}\" exists")
    torch.serialization.add_safe_globals([GptModelProgress])
    progress: GptModelProgress = torch.load(
      llm_file,
      map_location=device
    )
  else:
    progress = None  

  PATCHED_CONFIG = replace(
    GPT_CONFIG_124M, 
    drop_rate=0,
    transformer_drop_rate=0,
    attention_drop_rate=0,
  )

  model = ClassificationGptModel(PATCHED_CONFIG)

  if progress is not None:
    model.load_state_dict(progress.model_state_dict)

  model.to(device)

  optimAdamW = torch.optim.AdamW(
    params=model.parameters(),
    lr=5e-5,
    weight_decay=0.1,
  )

  if progress is not None:
    if progress.optim_state_dict is not None:
      optimAdamW.load_state_dict(progress.optim_state_dict)

  ml = ClassificationMachineLearning(
    model=model,
    device=device,
    optimizer=optimAdamW,
    train_loader=train_loader,
    valid_loader=valid_loader,
    evaluator=ClassificationModelEvaluator(
      model=model,
      train_loader=train_loader,
      valid_loader=valid_loader,
      device=device
    )
  )

  start_time = time.time()
  num_epochs = 5
  train_losses, valid_losses, train_accs, valid_accs, examples_seen = ml.train_model(
    num_epochs=num_epochs,
    eval_freq=50,
    eval_num_batches=5,
    on_epoch=lambda train_accuracy, valid_accuracy: handle_epoch(train_accuracy, valid_accuracy),
    on_batch=lambda *args, **kwargs: \
      ClassificationModelEvaluator.show_losses(*args, **kwargs)
  )

  end_time = time.time()
  execution_time_minutes = (end_time - start_time) / 60
  print(f"Training completed in {execution_time_minutes:.2f} minutes.")

  torch.save(
    GptModelProgress(
      model_state_dict=model.state_dict(),
      optim_state_dict=optimAdamW.state_dict(),
    ),
    llm_file
  )

  epochs_seen_tensor = torch.linspace(0, num_epochs, len(train_losses))
  examples_seen_tensor = torch.linspace(0, examples_seen, len(train_losses))
  ClassificationModelEvaluator.plot_values(
    epochs_seen=epochs_seen_tensor,
    examples_seen=examples_seen_tensor,
    train_values=train_losses,
    valid_values=valid_losses,
    label="Потери"
  )
  
  epochs_seen_tensor = torch.linspace(0, num_epochs, len(train_accs))
  examples_seen_tensor = torch.linspace(0, examples_seen, len(train_accs))
  ClassificationModelEvaluator.plot_values(
    epochs_seen=epochs_seen_tensor,
    examples_seen=examples_seen_tensor,
    train_values=train_accs,
    valid_values=valid_accs,
    label="Точность"
  )

if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  parser.add_argument("--llm-file", default=DEFAULT_LLM_FILE)
  parser.add_argument("--llm-train-source", default=DEFAULT_LLM_TRAIN_SOURCE)
  parser.add_argument("--llm-valid-source", default=DEFAULT_LLM_VALID_SOURCE)

  @dataclass
  class ArgsNamespace:
    llm_file: str
    llm_train_source: str
    llm_valid_source: str

  args = parser.parse_args(namespace=ArgsNamespace)

  if not os.path.exists(args.llm_train_source):
    raise ValueError(f"llm файл тренировочного источника не найден: {args.llm_train_source}")
  
  if not os.path.exists(args.llm_valid_source):
    raise ValueError(f"llm файл валидного источника не найден: {args.llm_valid_source}")

  run(
    llm_file=args.llm_file,
    llm_train_source=args.llm_train_source,
    llm_valid_source=args.llm_valid_source,
  )