from matplotlib import pyplot as plt
import torch


vocab = {
  "closer": 0,
  "every": 1,
  "effort": 2,
  "forward": 3,
  "inches": 4,
  "moves": 5,
  "pizza": 6,
  "toward": 7,
  "you": 8,
}

inverse_vocab = { v: k for k, v in vocab.items()}
print(inverse_vocab)

next_token_logits = torch.tensor(
  [4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79]
)

probas = torch.softmax(next_token_logits, dim=0)


def print_sampled_tokens_fast(probas):
  torch.manual_seed(123)
  count_map = {}

  for i in range(1_000):
    v = torch.multinomial(probas, num_samples=1).item()
    count_map[v] = count_map.get(v, 0) + 1
    
  for i, freq, in count_map.items():
    print(f"{freq} x {inverse_vocab[i]}")


def print_sampled_tokens(probas):
  torch.manual_seed(123)

  sample = [
    torch.multinomial(probas, num_samples=1).item() # сэмплирует индекс с вероятностью пропорциональной весам
    for i in range(1_000)
  ]
  # print("sample", sample)
  
  sampled_ids = torch.bincount(torch.tensor(sample))
  # print("sampled_ids", sampled_ids)

  for i, freq in enumerate(sampled_ids):
    print(f"{freq} x {inverse_vocab[i]}")


print("\n")
print_sampled_tokens_fast(probas)
print("\n")
print_sampled_tokens(probas)

def softmax_with_temperature(
  logits: torch.Tensor, 
  temperature: float
):
  scaled_logits = logits / temperature

  return torch.softmax(scaled_logits, dim=0)

# График зависимости выбора токена от температуры распределения:
# T < 1 — модель уверена, вероятность концентрируется на лучшем токене (жадный выбор)
# T = 1 — оригинальный softmax без изменений
# T > 1 — вероятности выравниваются, модель более творческая, но менее предсказуемая
# T → ∞ — все токены равновероятны (случайный выбор)
temperatures = [1, 0.1, 5]
scaled_probas = [
  softmax_with_temperature(next_token_logits, T) for T in temperatures
]
x = torch.arange(len(vocab))
bar_width = 0.15
fig, ax = plt.subplots(figsize=(5, 3))
for i, T in enumerate(temperatures):
  rects = ax.bar(x + i * bar_width, scaled_probas[i],
bar_width, label=f'Temperature = {T}')
ax.set_ylabel('Probability')
ax.set_xticks(x)
ax.set_xticklabels(vocab.keys(), rotation=90)
ax.legend()
plt.tight_layout()
plt.show()