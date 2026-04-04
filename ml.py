from typing import Optional, Protocol

from evaluator import ModelEvaluator
from dl import DataLoader
from torch.optim.optimizer import Optimizer
from torch.nn import Module
from torch import dtype

class OnBatchCallback(Protocol):
  def __call__(
    self, 
    epoch: int, 
    step: int,
    train_loss: float,
    valid_loss: float,
  ) -> None: ...

class OnEpochCallback(Protocol):
  def __call__(
    self, 
    device: Optional[dtype]
  ) -> None: ...

class MachineLearning():
  def __init__(
    self,
    model: Module,
    train_loader: DataLoader, 
    valid_loader: DataLoader, 
    optimizer: Optimizer, 
    device: dtype, 
  ):
    self.model = model
    self.train_loader = train_loader
    self.optimizer = optimizer
    self.device = device
    self.evaluator = ModelEvaluator(
      model=model,
      train_loader=train_loader,
      valid_loader=valid_loader,
      device=device
    )

  def train_model(
    self,
    num_epochs: int,
    eval_num_batches: int, 
    eval_freq: int, 
    on_epoch: Optional[OnEpochCallback] = None,
    on_batch: Optional[OnBatchCallback] = None
  ):
    train_losses: list[int] = []
    valid_losses: list[int] = []
    track_tokens_seen: list[int] = []

    tokens_seen = 0
    step = 0

    for epoch in range(num_epochs):
      self.model.train()

      for input_batch, target_batch in self.train_loader:
        self.optimizer.zero_grad() # Обнуляем графиенты потери после каждой итерации

        loss = self.evaluator.calc_loss_batch(
          input_batch=input_batch,
          target_batch=target_batch,
        )

        loss.backward() # Вычисление градиентов потерь

        self.optimizer.step() # Сердце обучения: обновление весов модели в соответствие с градиентами потерь

        tokens_seen += input_batch.numel()
        
        # Необязательный шаг оценки 
        if step % eval_freq == 0:
          train_loss, valid_loss = self.evaluator.evaluate_model(eval_num_batches)

          train_losses.append(train_loss)
          valid_losses.append(valid_loss)
          track_tokens_seen.append(tokens_seen)

          on_batch and on_batch(
            epoch=epoch,
            step=step,
            train_loss=train_loss,
            valid_loss=valid_loss,
          )
        
        step += 1
      
      on_epoch and on_epoch(self.device)
    
    return train_losses, valid_losses, track_tokens_seen
  
