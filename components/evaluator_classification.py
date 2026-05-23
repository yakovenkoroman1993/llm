
import torch
import matplotlib.pyplot as plt

from typing import Optional
from torch import Tensor, dtype
from components.dl import DataLoader
from components.evaluator import ModelEvaluator

class ClassificationModelEvaluator(ModelEvaluator):
  def calc_loss_loader(
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
  
  def calc_loss_batch(
    self,
    input_batch: Tensor,
    target_batch: Tensor,
    device: Optional[dtype] = None
  ) -> Tensor:
    input_batch = input_batch.to(device or self.device)
    target_batch = target_batch.to(device or self.device)
    
    logits: Tensor = self.model(input_batch)
    last_token_logits = logits[:, -1, :]
    
    loss = torch.nn.functional.cross_entropy(
      input=last_token_logits,
      target=target_batch,
    )

    return loss
  
  def calc_accuracy_loader(
    self,
    data_loader: DataLoader,
    device: dtype,
    num_batches=None
  ):
    self.model.eval()

    correct_predictions, num_examples = 0, 0

    # if len(data_loader) == 0:
    #   return float("nan")
    # elif num_batches is None:
    if num_batches is None:
      num_batches = len(data_loader)
    else:
      num_batches = min(num_batches, len(data_loader))
    
    for i, (input_batch, target_batch) in enumerate(data_loader):
      if (i < num_batches):

        input_batch = input_batch.to(device)
        target_batch = target_batch.to(device)

        with torch.no_grad():
          logits = self.model(input_batch)[:, -1, :]
        predicted_labels = torch.argmax(logits, dim=-1)

        num_examples += predicted_labels.shape[0]
        correct_predictions += (
          (predicted_labels == target_batch).sum().item()
        )
        
      else:
        break

    return correct_predictions / num_examples
  
  @staticmethod
  def plot_values(
    epochs_seen: Tensor, 
    examples_seen: Tensor, 
    train_values: list[float],
    valid_values: list[float],
    label: str
  ):
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epochs_seen, train_values, label=f"Training {label}")
    ax1.plot(
    epochs_seen, valid_values, linestyle="-.",
      label=f"Validation {label}"
    )
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(label.capitalize())
    ax1.legend()
    ax2 = ax1.twiny()
    ax2.plot(examples_seen, train_values, alpha=0)
    ax2.set_xlabel("Examples seen")
    fig.tight_layout()
    plt.savefig(f"{label}-plot.pdf")
    plt.show()
  