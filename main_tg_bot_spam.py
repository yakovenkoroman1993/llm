import os
import telebot
import tiktoken
import torch
import argparse

from dotenv import load_dotenv
from typeguard import typechecked
from aliases import LLM_SIZES, LlmSize
from cfg import GPT_MODEL_CONFIGS
from classes import GptModelProgress
from telebot.types import Message
from dataclasses import dataclass, replace
from components.gpt_agent_classification import ClassificationGptModelAgent
from components.gpt_model_classification import ClassificationGptModel

load_dotenv()

DEFAULT_LLM_FILE = "gpt2_124m_spam.pth"
DEFAULT_LLM_SIZE = "124M"

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

if not TELEGRAM_TOKEN:
  raise ValueError("TELEGRAM_TOKEN не найден в .env файле")

@typechecked
def run(
  llm_size: LlmSize = DEFAULT_LLM_SIZE,
  llm_file = DEFAULT_LLM_FILE
):
  tgBot = telebot.TeleBot(TELEGRAM_TOKEN)

  device = "cpu"
    
  tokenizer = tiktoken.get_encoding("gpt2")

  if os.path.exists(llm_file):
    print(f"LLM \"{llm_file}\" exists")
    torch.serialization.add_safe_globals([GptModelProgress])
    progress: GptModelProgress = torch.load(
      llm_file,
      map_location=device
    )
  else:
    raise RuntimeError(f"Expected pre‑trained LLM file but not found: {llm_file}")

  PATCHED_CONFIG = replace(
    GPT_MODEL_CONFIGS[llm_size], 
    drop_rate=0,
    transformer_drop_rate=0,
    attention_drop_rate=0,
  )
  
  model = ClassificationGptModel(PATCHED_CONFIG)
  # print(model)
  
  if progress is not None:
    model.load_state_dict(progress.model_state_dict)

  model.to(device)

  model.eval()

  agent = ClassificationGptModelAgent(
    model=model,
    device=device,
    tokenizer=tokenizer
  )

  def chat_ai(user_message: str):
    try:
      answer = agent.send_message(user_message)

      return answer \
        .replace("\n", " ")  \
        .replace(user_message, "")  
      
    except Exception as e:
      print(f"Ошибка при обращении к ИИ: {e}")

      return (
        "Извините, произошла ошибка при обработке запроса."
        "Попробуйте позже или свяжитесь с оператором."
      )

  @tgBot.message_handler(func=lambda message: True)
  def handle_message(message: Message):
    user_text: str = message.text
    tgBot.send_chat_action(message.chat.id, "typing")
    response_text = chat_ai(user_text)
    tgBot.send_message(message.chat.id, response_text)

  print("Бот с ИИ запущен и ожидает сообщений...")
  print("Для остановки нажмите Ctrl+C")
  tgBot.polling(none_stop=True)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  parser.add_argument("--llm-file", default=DEFAULT_LLM_FILE)
  parser.add_argument("--llm-size", default=DEFAULT_LLM_SIZE, choices=LLM_SIZES)

  @dataclass
  class ArgsNamespace:
    llm_size: str
    llm_file: str

  args = parser.parse_args(namespace=ArgsNamespace)

  run(
    llm_file=args.llm_file,
    llm_size=args.llm_size
  )