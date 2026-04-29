
import math
import matplotlib.pyplot as plt
import torch

from typing import Optional
from matplotlib.ticker import MaxNLocator
from torch.nn import Module
from torch import Tensor, dtype
from components.dl import DataLoader

class ModelEvaluator():
  def __init__(
    self,
    model: Module,
    train_loader: DataLoader, 
    valid_loader: DataLoader, 
    device: dtype, 
  ):
    self.model = model
    self.train_loader = train_loader
    self.valid_loader = valid_loader
    self.device = device

  def evaluate_model(
    self,
    num_batches: int,
    device: Optional[dtype] = None,
  ):
    self.model.eval() # model.train(False) - выкл обучения, отключение отсева Dropout

    with torch.no_grad():
      train_loss = self.__calc_loss_loader(
        data_loader=self.train_loader,
        device=device or self.device,
        num_batches=num_batches
      )
      valid_loss = self.__calc_loss_loader(
        data_loader=self.valid_loader,
        device=device or self.device,
        num_batches=num_batches
      )

    self.model.train() # вкл обратно обучение

    return train_loss, valid_loss

  def calc_loss_batch(
    self,
    input_batch: Tensor,
    target_batch: Tensor,
    device: Optional[dtype] = None
  ) -> Tensor:
    input_batch = input_batch.to(device or self.device)
    target_batch = target_batch.to(device or self.device)
    
    logits: Tensor = self.model(input_batch)
    
    loss = torch.nn.functional.cross_entropy(
      input=logits.flatten(0, 1),
      target=target_batch.flatten(),
    )

    return loss

  def __calc_loss_loader(
    self,
    data_loader: DataLoader,
    device: dtype,
    num_batches=None
  ):
    total_loss = 0

    if len(data_loader) == 0:
      return float("nan")
    elif num_batches is None:
      num_batches = len(data_loader)
    else:
      num_batches = min(num_batches, len(data_loader))
    
    for i, (input_batch, target_batch) in enumerate(data_loader):
      if (i < num_batches):
        loss = self.calc_loss_batch(
          input_batch=input_batch,
          target_batch=target_batch,
          device=device or self.device,
        )

        total_loss += loss.item()
      else:
        break

    return total_loss / num_batches
  
  staticmethod
  def show_losses(
    epoch: int,
    step: int,
    train_loss: float,
    valid_loss: float,
  ):
    # Оценка затруднения. 
    # Модель не уверена в том, какой из {train_perplexity} токенов в словаре следует сгенерировать в качестве следующего токена
    train_perplexity = math.exp(train_loss) 
    valid_perplexity = math.exp(valid_loss)

    print(
      f"Epoch {epoch+1} (Step {step:06d}): "
      f"\n"
      f"Train loss = {train_loss:.3f}, "
      f"perplexity = {train_perplexity:.0f}"
      f"\n"
      f"Valid loss = {valid_loss:.3f}, "
      f"perplexity = {valid_perplexity:.0f}"
      f"\n"
    )
  
  @staticmethod
  def plot_losses(
    train_losses: list[int],
    valid_losses: list[int],
    tokens_seen: list[int],
  ):
    epochs_seen = torch.linspace(0, 10, len(train_losses))

    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(
      epochs_seen, valid_losses, linestyle="-.", label="Validation loss"
    )
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens seen")
    fig.tight_layout()
    plt.show()