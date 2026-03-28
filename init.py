import tiktoken
from cfg import GptModelConfig
import torch
from torch import Tensor
from gpt import GptModel

with open("the-verdict.txt", "r", encoding="utf-8") as f:
  raw_text = f.read()


# # 1.0 Создаем итератор для пакетов с токенами: 1 пакет = 8 последовательностей по 4 токена
# max_length = 4
# data_loader = dl.create(
#   raw_text,
#   batch_size=8,
#   max_length=max_length,
#   stride=max_length,
# )

# data_iter = iter(data_loader)
# inputs, targets = next(data_iter)

#### Токенизация
tokenizer = tiktoken.get_encoding("gpt2")
tokenList: list[Tensor] = []
txt1 = "Every effort moves you"
txt2 = "Every day holds a"
tokenList.append(torch.tensor(tokenizer.encode(txt1)))
tokenList.append(torch.tensor(tokenizer.encode(txt2)))

batch: Tensor = torch.stack(tokenList, dim=0) # list[Tensor] -> Tensor

GPT_CONFIG_124M = GptModelConfig(
  vocab_size=50257,
  context_length=1024,
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


torch.manual_seed(123)
model = GptModel(GPT_CONFIG_124M)
out = model(batch)
print("Input batch:\n", batch)
print("\nOutput shape:", out.shape)
print(out)

# Итеративная генерация текста с получаемых логитов
def gen_text_simple(
  model: GptModel,
  idx: Tensor,
  max_new_tokens: int,
  context_size: int
):
  for _ in range(max_new_tokens):
    idx_cond = idx[:, -context_size:]

    with torch.no_grad():
      logits = model(idx_cond)
    
    logits = logits[:, -1, :]
    probas = torch.softmax(logits, dim=-1)
    idx_next = torch.argmax(probas, dim=-1, keepdim=True)
    idx = torch.concat((idx, idx_next), dim=-1)
  
  return idx

start_context = "Hello, I am"
encoded = tokenizer.encode(start_context)
print("encoded:", encoded)
encoded_tensor = torch.tensor(encoded).unsqueeze(0) # list to matrix

model.eval() # отключение отсева nn.Dropout

out = gen_text_simple(
  model=model,
  idx=encoded_tensor,
  max_new_tokens=6,
  context_size=GPT_CONFIG_124M.context_length
)

print("Output:", out)
print("Output length:", len(out[0]))

decoded = tokenizer.decode(out.squeeze(0).tolist())
print("decoded", decoded)
