from typing import Optional, Protocol
from components.ml import MachineLearning, OnBatchCallback

class OnEpochCallback(Protocol):
  def __call__(
    self, 
    train_accuracy: float,
    valid_accuracy: float
  ) -> None: ...

class ClassificationMachineLearning(MachineLearning):
  def train_model(
    self,
    num_epochs: int,
    eval_num_batches: int, 
    eval_freq: int, 
    on_epoch: Optional[OnEpochCallback] = None,
    on_batch: Optional[OnBatchCallback] = None
  ):
    train_losses: list[float] = []
    valid_losses: list[float] = []
    
    examples_seen = 0

    train_accs: list[float] = [] 
    valid_accs: list[float] = []

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

        examples_seen += input_batch.shape[0] # Отслеживает примеры вместо токенов
        
        # Необязательный шаг оценки 
        if step % eval_freq == 0:
          train_loss, valid_loss = self.evaluator.evaluate_model(eval_num_batches)

          train_losses.append(train_loss)
          valid_losses.append(valid_loss)

          on_batch and on_batch(
            epoch=epoch,
            step=step,
            train_loss=train_loss,
            valid_loss=valid_loss,
          )
        
        step += 1
      
      train_accuracy = self.evaluator.calc_accuracy_loader(
        data_loader=self.train_loader,
        device=self.device,
        num_batches=eval_num_batches,
      )
      
      valid_accuracy = self.evaluator.calc_accuracy_loader(
        data_loader=self.valid_loader,
        device=self.device,
        num_batches=eval_num_batches,
      )

      train_accs.append(train_accuracy)
      valid_accs.append(valid_accuracy)

      on_epoch and on_epoch(train_accuracy, valid_accuracy)
    
    return train_losses, valid_losses, train_accs, valid_accs, examples_seen
  
