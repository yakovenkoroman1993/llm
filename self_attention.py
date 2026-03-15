import torch
from torch import Tensor

# softmax(2D x 2D.T) х 2D
def gen_inputs_context(inputs: Tensor):
  return torch.softmax(inputs @ inputs.T, dim=-1) @ inputs
  """ 
  Полная реализация по шагам:
  contexts = torch.empty(inputs.shape)

  for i, target_input in enumerate(inputs):
    # Вычислим коэффициенты внимаенния для текущего вложения - скалярное произведение: w_j = x_j * x_i
    attention_scores= torch.empty(inputs.shape[0])
    for j, input in enumerate(inputs):
      attention_scores[j] = torch.dot(input, target_input)

    # Нормализуем значения вектора, чтобы их сумма равнялась 1 (softmax)
    attention_weights = torch.softmax(attention_scores, dim=0)

    # Получим контекстный вектор для текущего вложения: сумма x_j * w_j 
    context = torch.zeros(target_input.shape)
    for j, input in enumerate(inputs):
      context += input * attention_weights[j]    
    
    contexts[i] = context

  return contexts
   
  """
  

class SelfAttention_v1(torch.nn.Module):
  def __init__(self, d_in: int, d_out: int):
    super().__init__()

    self.W_query = torch.nn.Parameter(torch.rand(d_in, d_out))
    self.W_key = torch.nn.Parameter(torch.rand(d_in, d_out))
    self.W_value = torch.nn.Parameter(torch.rand(d_in, d_out))

  def forward(self, inputs: Tensor):
    queries = inputs @ self.W_query # Q
    keys = inputs @ self.W_key      # K
    values = inputs @ self.W_value  # V
    attention_scores = queries @ keys.T

    # масштабированние скалярного произведения чтобы избежать нулевых градиентов
    attention_weights = torch.softmax(attention_scores / keys.shape[-1]**0.5, dim=-1) 
    print("ATTENTION WEIGHTS", attention_weights)
    
    return attention_weights @ values

class SelfAttention_v2(torch.nn.Module):
  def __init__(self, d_in: int, d_out: int, bias=False):
    super().__init__()
    # self.W_query = torch.nn.Linear(d_in, d_out, bias).weight.T
    self.W_query = torch.nn.Linear(d_in, d_out, bias)
    # self.W_key = torch.nn.Linear(d_in, d_out, bias).weight.T
    self.W_key = torch.nn.Linear(d_in, d_out, bias)
    # self.W_value = torch.nn.Linear(d_in, d_out, bias).weight.T
    self.W_value = torch.nn.Linear(d_in, d_out, bias)

  def forward(self, inputs: Tensor):
    queries = self.W_query(inputs) # Q
    keys = self.W_key(inputs)      # K
    values = self.W_value(inputs)  # V
    attention_scores = queries @ keys.T

    # масштабированние скалярного произведения чтобы избежать нулевых градиентов
    attention_weights = torch.softmax(attention_scores / keys.shape[-1]**0.5, dim=-1) 
    print("ATTENTION WEIGHTS", attention_weights)
    
    return attention_weights @ values
