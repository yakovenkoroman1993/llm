import os
import telebot
import tiktoken
import torch
import argparse

from dotenv import load_dotenv
from typeguard import typechecked
from aliases import LLM_SIZES, LlmSize
from cfg import GPT_MODEL_CONFIGS
from components.gpt_agent import GptModelAgent
from components.gpt_model import GptModel
from classes import GptModelProgress
from telebot.types import Message
from dataclasses import dataclass, replace

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
if not TELEGRAM_TOKEN:
  raise ValueError("TELEGRAM_TOKEN не найден в .env файле")

@typechecked
def run(
  llm_size: LlmSize,
  llm_file: str
):
  tgBot = telebot.TeleBot(TELEGRAM_TOKEN)

  device = "cpu"
    
  tokenizer = tiktoken.get_encoding("gpt2")

  torch.serialization.add_safe_globals([GptModelProgress])
  progress: GptModelProgress = torch.load(
    llm_file,
    map_location=device
  )

  gpt_cfg = GPT_MODEL_CONFIGS[llm_size]
  print("GPT_CONFIG", gpt_cfg)
  model = GptModel(gpt_cfg)
  
  if progress is not None:
    model.load_state_dict(progress.model_state_dict)

  model.to(device)

  agent = GptModelAgent(
    model=model,
    tokenizer=tokenizer,
    device=device,
  )

  model.eval()

  agent = GptModelAgent(
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

  parser.add_argument("--llm-file", required=True)
  parser.add_argument("--model-size", required=True, choices=LLM_SIZES)

  @dataclass
  class ArgsNamespace:
    llm_size: str
    llm_file: str

  args = parser.parse_args(namespace=ArgsNamespace)

  run(
    llm_file=args.llm_file,
    llm_size=args.llm_size
  )