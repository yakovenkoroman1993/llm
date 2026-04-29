import torch
from torch import Tensor
from classes import GptModelConfig
import matplotlib.pyplot as plt

class ActivationsGraph():
  def graph(self, x: Tensor):
    gelu, relu = GELU(), torch.nn.ReLU()

    y_gelu, y_relu = gelu(x), relu(x)

    plt.figure(figsize=(8, 3))

    for i, (y, label) in enumerate(zip([y_gelu, y_relu], ["GELU", "ReLU"]), 1):
      plt.subplot(1, 2, i)
      plt.plot(x, y)
      plt.title(f"{label} activation function")
      plt.xlabel("x")
      plt.ylabel(f"{label}(x)")
      plt.grid(True)

    plt.tight_layout()
    plt.show()

class GELU(torch.nn.Module):
  def __init__(self):
    super().__init__()
  
  def __call__(self, x: Tensor) -> Tensor:
    return super().__call__(x)
  
  def forward(self, x: Tensor):
    return 0.5 * x * (1 + torch.tanh(
      torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))
    ))
  
class FeedForward(torch.nn.Module):
  def __init__(self, cfg: GptModelConfig):
    super().__init__()

    self.layers = torch.nn.Sequential(
      torch.nn.Linear(cfg.embedding_dim, 4 * cfg.embedding_dim),
      GELU(),
      torch.nn.Linear(4 * cfg.embedding_dim, cfg.embedding_dim),
    )

  def __call__(self, x: Tensor) -> Tensor:
    return super().__call__(x)
  
  def forward(self, x):
    return self.layers(x)


class ExampleDeepNeuralNetwork(torch.nn.Module):
  def __init__(
    self, 
    layer_sizes: list[int], 
    use_shortcut: bool
  ):
    super().__init__()

    self.use_shortcut = use_shortcut

    self.layers = torch.nn.ModuleList([
      torch.nn.Sequential(
        torch.nn.Linear(layer_sizes[i], layer_sizes[i + 1]),
        GELU()
      )
      for i in range(len(layer_sizes) - 1)
    ])


  def __call__(self, x: Tensor) -> Tensor:
    return super().__call__(x)
  
  def forward(self, x: Tensor):
    for layer in self.layers:
      layer_output = layer(x)
      if self.use_shortcut and layer_output.shape == x.shape:
        x = x + layer_output
      else:
        x = layer_output

    return x