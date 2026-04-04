from torch import Tensor
import torch

# Реализация torch.nn.functional.cross_entropy
def cross_entropy_manually(
  model: torch.nn.Module,
  inputs: Tensor,
  targets: Tensor,
):
  with torch.no_grad():
    logits = model(inputs)

  probas = torch.softmax(logits, dim=-1)

  target_probas_1 = probas[0, [0, 1, 2], targets[0]] # вытаскиваем веса для правильных токенов
  target_probas_2 = probas[1, [0, 1, 2], targets[1]]

  log_probas = torch.log(
    torch.cat((target_probas_1, target_probas_2))
  )

  avg_log_probas = log_probas.mean()
  loss = -avg_log_probas # результат перекрестной энтропии

  return loss
