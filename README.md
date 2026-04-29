# GPT-2 — Обучение и Telegram-бот

Проект реализует GPT-2 с нуля на PyTorch. Поддерживает три режима: импорт весов из OpenAI, дообучение на своих данных и запуск Telegram-бота на основе модели.

---

## Требования

```bash
pip install torch tiktoken typeguard python-dotenv pyTelegramBotAPI numpy
```

Создайте файл `.env` в корне проекта:

```env
TELEGRAM_TOKEN=your_telegram_bot_token
```

---

## Режимы запуска

Все режимы запускаются через единую точку входа `main.py`.

### 1. Импорт весов GPT-2 от OpenAI

Загружает предобученные веса GPT-2 и сохраняет их в файл модели.

```bash
python main.py --mode import --llm-file model.pth --llm-size 124M
```

| Аргумент | Описание |
|---|---|
| `--llm-size` | Размер модели: `124M`, `355M`, `774M`, `1558M` |
| `--llm-file` | Путь для сохранения файла модели |

---

### 2. Обучение на своих данных

Дообучает модель на текстовом файле. Поддерживает продолжение с чекпоинта.

```bash
python main.py --mode train --llm-file model.pth
```

По умолчанию использует файл `the-verdict.txt` как источник данных. Файл источника можно переопределить, запустив `main_train.py` напрямую:

```bash
python main_train.py --llm-file model.pth --llm-source my_text.txt
```

Во время обучения каждую эпоху модель генерирует продолжение фразы `"Every effort moves you"` — так можно наблюдать прогресс. По завершении строится график train/valid loss.

---

### 3. Telegram-бот

Запускает бота, который отвечает на сообщения с помощью загруженной модели.

```bash
python main.py --mode tg-bot --llm-file model.pth --llm-size 124M
```

Бот принимает текстовое сообщение и возвращает продолжение, сгенерированное моделью.

---

## Структура проекта

```
.
├── main.py                  # Точка входа, роутинг по режимам
├── main_import.py           # Импорт весов из OpenAI GPT-2
├── main_train.py            # Цикл обучения
├── main_tg_bot.py           # Telegram-бот
├── cfg.py                   # Конфиги моделей GPT_MODEL_CONFIGS
├── aliases.py               # Типы LlmSize, LLM_SIZES
├── classes.py               # GptModelProgress (датакласс чекпоинта)
├── components/
│   ├── gpt_model.py         # Архитектура GptModel
│   ├── gpt_agent.py         # GptModelAgent (генерация текста)
│   ├── transformer.py       # Блок трансформера
│   ├── causal_attention.py  # MultiHeadAttention
│   ├── ml.py                # MachineLearning (цикл train/eval)
│   ├── evaluator.py         # ModelEvaluator (метрики, графики)
│   └── dl.py                # DataLoader для текстовых данных
├── import_gpt2/
│   └── gpt_download.py      # Загрузка весов GPT-2 от OpenAI
├── the-verdict.txt          # Пример обучающих данных
└── .env                     # TELEGRAM_TOKEN (не коммитить)
```

---

## Типичный воркфлоу

```bash
# 1. Импортировать предобученные веса
python main.py --mode import --llm-file gpt2_124m.pth --llm-size 124M

# 2. Дообучить на своих данных (опционально)
python main.py --mode train --llm-file gpt2_124m.pth

# 3. Запустить бота
python main.py --mode tg-bot --llm-file gpt2_124m.pth --llm-size 124M
```