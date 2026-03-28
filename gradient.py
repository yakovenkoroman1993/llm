
import torch

from torch import Tensor
from gelu import ExampleDeepNeuralNetwork

def print_gradients(model: torch.nn.Module, x: Tensor):
  output = model(x)
  target = torch.tensor([[0.]])

  loss = torch.nn.MSELoss()
  loss: Tensor = loss(output, target)

  loss.backward()

  for name, param in model.named_parameters():
    if 'weight' in name:
      print(f'{name} has gradient mean of {param.grad.abs().mean().item()}')

# Наглядная проблема затухания градиентов (производных) вовремя обратного прохода модели
def show_gradients_fadding():
  layer_sizes = [3, 3, 3, 3, 3, 1]

  sample_input = torch.tensor([[1., 0., -1.]])

  model_without_shortcut = ExampleDeepNeuralNetwork(
    layer_sizes=layer_sizes,
    use_shortcut=False
  )
  
  model_with_shortcut = ExampleDeepNeuralNetwork(
    layer_sizes=layer_sizes,
    use_shortcut=True
  )

  print_gradients(model_without_shortcut, sample_input)
  print_gradients(model_with_shortcut, sample_input)