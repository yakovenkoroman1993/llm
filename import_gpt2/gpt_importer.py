from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
from torch import Tensor
import torch

from components.causal_attention import MultiHeadAttention
from components.gpt_model import GptModel
from components.transformer import Transformer

def compare_shapes(left: Tensor, right: Tensor):
  if left.shape != right.shape:
    raise ValueError(
      f"Shape mismatch. Left: {left.shape}, ""Right: {right.shape}"
    )

def in_param(input):
  return torch.nn.Parameter(torch.tensor(input))

LlmLayer = Literal["transformer", "final_norm", "out_head"]

@dataclass
class GptImportOptions:
  excluded_layers: list[LlmLayer] = field(default_factory=list)

def load_weights_into_gpt(
  gpt: GptModel, 
  params,
  options: Optional[GptImportOptions] = None
):
  options = options or GptImportOptions()
  
  compare_shapes(gpt.position_embedding.weight, params["wpe"])
  gpt.position_embedding.weight = in_param(params["wpe"])

  compare_shapes(gpt.token_embedding.weight, params["wte"])
  gpt.token_embedding.weight = in_param(params["wte"])
  
  if "transformer" not in options.excluded_layers:
    for b in range(len(params["blocks"])):
      transformer: Transformer = gpt.transformer_blocks[b]

      # веса внимания QKV
      q_w, k_w, v_w = np.split(
        params["blocks"][b]["attn"]["c_attn"]["w"], 3, axis=-1
      )
      
      attention: MultiHeadAttention = transformer.attention

      compare_shapes(attention.W_query.weight, q_w.T)
      attention.W_query.weight = in_param(q_w.T)
    
      compare_shapes(attention.W_key.weight, k_w.T)
      attention.W_key.weight = in_param(k_w.T)

      compare_shapes(attention.W_value.weight, v_w.T)
      attention.W_value.weight = in_param(v_w.T)

      q_b, k_b, v_b = np.split(
        params["blocks"][b]["attn"]["c_attn"]["b"], 3, axis=-1
      )

      compare_shapes(attention.W_query.bias, q_b)
      attention.W_query.bias = in_param(q_b)
      
      compare_shapes(attention.W_key.bias, k_b)
      attention.W_key.bias = in_param(k_b)
      
      compare_shapes(attention.W_value.bias, v_b)
      attention.W_value.bias = in_param(v_b)
      
      # out
      c_proj_w = params["blocks"][b]["attn"]["c_proj"]["w"].T
      compare_shapes(attention.out.weight, c_proj_w)
      attention.out.weight = in_param(c_proj_w)
      
      c_proj_b = params["blocks"][b]["attn"]["c_proj"]["b"]
      compare_shapes(attention.out.bias, c_proj_b)
      attention.out.bias = in_param(c_proj_b)

      # FeedForward
      ff = transformer.ff 

      ff_first_layer: torch.nn.Linear = ff.layers[0]

      c_fc_w = params["blocks"][b]["mlp"]["c_fc"]["w"].T
      compare_shapes(ff_first_layer.weight, c_fc_w)
      ff_first_layer.weight = in_param(c_fc_w)
      
      c_fc_b = params["blocks"][b]["mlp"]["c_fc"]["b"]
      compare_shapes(ff_first_layer.bias, c_fc_b)
      ff_first_layer.bias = in_param(c_fc_b)

      ff_third_layer: torch.nn.Linear = ff.layers[2]

      c_proj_w = params["blocks"][b]["mlp"]["c_proj"]["w"].T
      compare_shapes(ff_third_layer.weight, c_proj_w)
      ff_third_layer.weight = in_param(c_proj_w)

      c_proj_b = params["blocks"][b]["mlp"]["c_proj"]["b"]
      compare_shapes(ff_third_layer.bias, c_proj_b)
      ff_third_layer.bias = in_param(c_proj_b)
      
      # norm слои
      norm1 = transformer.norm1

      ln_1_g = params["blocks"][b]["ln_1"]["g"]
      compare_shapes(norm1.scale, ln_1_g)
      norm1.scale = in_param(ln_1_g)
      
      ln_1_b = params["blocks"][b]["ln_1"]["b"]
      compare_shapes(norm1.shift, ln_1_b)
      norm1.shift = in_param(ln_1_b)
      
      norm2 = transformer.norm2

      ln_2_g = params["blocks"][b]["ln_2"]["g"]
      compare_shapes(norm2.scale, ln_2_g)
      norm2.scale = in_param(ln_2_g)
      
      ln_2_b = params["blocks"][b]["ln_2"]["b"]
      compare_shapes(norm2.shift, ln_2_b)
      norm2.shift = in_param(ln_2_b)

  # слои финальной нормализация
  if "final_norm" not in options.excluded_layers:
    compare_shapes(gpt.final_norm.scale, params["g"])
    gpt.final_norm.scale = in_param(params["g"])

  
    compare_shapes(gpt.final_norm.shift, params["b"])
    gpt.final_norm.shift = in_param(params["b"])

  if "out_head" not in options.excluded_layers:
    compare_shapes(gpt.out_head.weight, params["wte"])
    gpt.out_head.weight = in_param(params["wte"])