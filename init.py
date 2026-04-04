from matplotlib import pyplot as plt
import tiktoken
from cfg import GptModelConfig
import torch
import dl
from encoder import Encoder
from evaluator import ModelEvaluator
from gpt_model import GptModel
from gpt_agent import GptModelAgent
from ml import MachineLearning

with open("the-verdict.txt", "r", encoding="utf-8") as f:
  raw_text = f.read()

#### Токенизация
tokenizer = tiktoken.get_encoding("gpt2")

# tokenList: list[Tensor] = []
# txt1 = "Every effort moves you"
# txt2 = "Every day holds a"
# tokenList.append(torch.tensor(tokenizer.encode(txt1)))
# tokenList.append(torch.tensor(tokenizer.encode(txt2)))
# batch: Tensor = torch.stack(tokenList, dim=0) # list[Tensor] -> Tensor

GPT_CONFIG_124M = GptModelConfig(
  vocab_size=50257,
  # context_length=1024,
  context_length=256, # чтобы уменьшить нагрузку при обучении
  embedding_dim=768,
  num_heads=12,
  num_layers=12,
  drop_rate=0.1,
  transformer_drop_rate=0.1,
  attention_drop_rate=0.1,
  qkv_bias=False
)

# ag = ActivationsGraph()
# ag.graph(
#   x=torch.linspace(-3, 3, 100)
# )


# inputs = torch.tensor([
#   [16833, 3626, 6100],  # ["every effort moves",
#   [40, 1107, 588],      # "I really like"]
# ])

# targets = torch.tensor([
#   [3626, 6100, 345 ],   # [" effort moves you",
#   [1107, 588, 11311],   # " really like chocolate"]
# ])

# with torch.no_grad():
#   logits = model(inputs)

# logits_flat = logits.flatten(0, 1)
# targets_flat = targets.flatten()
# loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
# # print("loss", loss)


# ***

train_ratio = 0.9
split_position = int(train_ratio * len(raw_text))
train_data = raw_text[:split_position]
valid_data = raw_text[split_position:]

# Создаем итератор для пакетов с токенами: 1 пакет = 2 последовательности по 256 токена
train_loader = dl.create(
  txt=train_data,
  batch_size=2,
  max_length=GPT_CONFIG_124M.context_length,
  stride=GPT_CONFIG_124M.context_length,
  drop_last=True,
  shuffle=True,
)

valid_loader = dl.create(
  txt=valid_data,
  batch_size=2,
  max_length=GPT_CONFIG_124M.context_length,
  stride=GPT_CONFIG_124M.context_length,
  drop_last=False,
  shuffle=False,
)

torch.manual_seed(123)

model = GptModel(GPT_CONFIG_124M)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# device = "cpu"
model.to(device)

optimAdamW = torch.optim.AdamW(
  params=model.parameters(),
  lr=0.0004,
  weight_decay=0.1,
)

ml = MachineLearning(
  model=model,
  device=device,
  optimizer=optimAdamW,
  train_loader=train_loader,
  valid_loader=valid_loader,
)

agent = GptModelAgent(
  model=model,
  tokenizer=tokenizer,
  device=device,
)

def hanlde_epoch(device: torch.dtype):
  answer = agent \
    .send_message("Every effort moves you", device) \
    .replace('\n', ' ')

  print(
    f"\n***\n"
    f"Gpt Agent Response: {answer}"
    f"\n***\n" 
  ),

train_losses, valid_losses, tokens_seen = ml.train_model(
  num_epochs=10,
  eval_freq=5,
  eval_num_batches=5,
  on_epoch=lambda device: hanlde_epoch(device),
  on_batch=lambda *args, **kwargs: \
    ModelEvaluator.show_losses(*args, **kwargs)
)

ModelEvaluator.plot_losses(
  train_losses=train_losses,
  valid_losses=valid_losses,
  tokens_seen=tokens_seen
)

# model.eval()

# agent = GptModelAgent(
#   model=model,
#   device=device,
#   tokenizer=tokenizer
# )
# encoder = Encoder(tokenizer)

# answer = agent.send_message("Every effort moves you")

# print("Output text:\n", answer.replace('\n', ' '))

