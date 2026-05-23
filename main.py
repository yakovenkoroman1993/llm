import argparse

from dataclasses import dataclass
from typing import Literal, Optional
from aliases import LLM_SIZES, LlmSize

parser = argparse.ArgumentParser()

parser.add_argument("--mode", required=True, choices=[
  "train", 
  "train-spam", 
  "import", 
  "import-spam", 
  "tg-bot", 
  "tg-bot-spam",
])
parser.add_argument("--llm-file", required=True)
parser.add_argument("--llm-size", choices=LLM_SIZES)

@dataclass
class ArgsNamespace:
  mode: Literal[
    "train", 
    "train-spam", 
    "import", 
    "import-spam",
    "tg-bot", 
    "tg-bot-spam"
  ]
  llm_file: str
  llm_size: Optional[LlmSize]

args = parser.parse_args(namespace=ArgsNamespace)

if (args.mode == "import" or args.mode == "tg-bot") \
  and not args.llm_size:
  parser.error("--llm-size обязателен для режима import")


if args.mode == "train":
  from main_train import run
  run(llm_file=args.llm_file)

elif args.mode == "train-spam":
  from main_train_spam import run
  run(llm_file=args.llm_file)

elif args.mode == "import":
  from main_import import run
  run(llm_file=args.llm_file, llm_size=args.llm_size)

elif args.mode == "import-spam":
  from main_import_spam import run
  run(llm_file=args.llm_file, llm_size=args.llm_size)

elif args.mode == "tg-bot":
  from main_tg_bot import run
  run(llm_file=args.llm_file, llm_size=args.llm_size)

elif args.mode == "tg-bot-spam":
  from main_tg_bot_spam import run
  run(llm_file=args.llm_file, llm_size=args.llm_size)
