import os
import torch
import argparse

from typeguard import typechecked
from aliases import LLM_SIZES, LlmSize
from cfg import GPT_MODEL_CONFIGS
from components.gpt_model_classification import ClassificationGptModel
from import_gpt2.gpt_importer import GptImportOptions, load_weights_into_gpt
from import_gpt2.gpt_download import load_gpt2
from classes import GptModelProgress
from dataclasses import dataclass

DEFAULT_MODEL_SIZE = "124M"
DEFAULT_LLM_FILE = "gpt2_124m_spam.pth"

@typechecked
def run(
  llm_size: LlmSize,
  llm_file: str
):
  gpt_cfg = GPT_MODEL_CONFIGS[llm_size]
  print(gpt_cfg)
  gpt = ClassificationGptModel(gpt_cfg)

  gpt.eval()

  # START IMPORT
  _, params = load_gpt2(
    model_size=llm_size, 
    models_dir="gpt2",
    base_dir=os.path.dirname(os.path.abspath(__file__))
  )

  load_weights_into_gpt(
    gpt, 
    params, 
    options=GptImportOptions(
      excluded_layers=["out_head"]
    )
  )

  torch.save(
    GptModelProgress(
      model_state_dict=gpt.state_dict(),
      optim_state_dict=None,
    ),
    llm_file
  )

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--llm-size", default=DEFAULT_MODEL_SIZE, choices=LLM_SIZES)
  parser.add_argument("--llm-file", default=DEFAULT_LLM_FILE)

  @dataclass
  class ArgsNamespace:
    llm_size: str
    llm_file: str

  args = parser.parse_args(namespace=ArgsNamespace)

  run(
    llm_size=args.llm_size,
    llm_file=args.llm_file
  )