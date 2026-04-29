import torch

from components.gelu import ActivationsGraph


ag = ActivationsGraph()

ag.graph(
  x=torch.linspace(-3, 3, 100)
)
