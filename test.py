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
from classes import GptModelProgress
from telebot.types import Message
from dataclasses import dataclass, replace
from components.gpt_agent_classification import ClassificationGptModelAgent
from components.gpt_model import GptModel
from components.gpt_model_classification import ClassificationGptModel

load_dotenv()


tokenizer = tiktoken.get_encoding("gpt2")

device = "cpu"
llm_file = "gpt2_124m_spam.pth"
llm_size = "124M"

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

print("GPT_CONFIG", PATCHED_CONFIG)
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

def classify_review(
  text, 
  model: GptModel, 
  tokenizer,
  device,
  max_length=None, 
  pad_token_id=50256
):
  model.eval()
  input_ids = tokenizer.encode(text)
  supported_context_length = model.position_embedding.weight.shape[1]
  input_ids = input_ids[:min(
    max_length, supported_context_length
  )]
  input_ids += [pad_token_id] * (max_length - len(input_ids))
  input_tensor = torch.tensor(
    input_ids, device=device
  ).unsqueeze(0)

  with torch.no_grad():
    logits = model(input_tensor)[:, -1, :]
  
  predicted_label = torch.argmax(logits, dim=-1).item()

  print("spam" if predicted_label == 1 else "not spam")

text_1 = "You are a winner you have been specially selected to receive $1000 cash or a $2000 award."
text_2 = "Hey, just wanted to check if we're still on for dinner tonight? Let me know!"

classify_review(
  model=model,
  device=device,
  max_length=120,
  tokenizer=tokenizer,
  # text=text_1,
  text=text_2,
)

print("AGENT", agent.send_message(text_2))