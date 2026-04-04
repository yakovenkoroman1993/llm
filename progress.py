
from dataclasses import dataclass
from typing import Any, Mapping
from torch.optim.optimizer import StateDict as OptimStateDict

@dataclass
class GptModelProgress:
  model_state_dict: Mapping[str, Any]
  optim_state_dict: OptimStateDict