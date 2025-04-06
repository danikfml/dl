from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import logging


# Настройка логов
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Определение устройства
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    model = AutoModelForCausalLM.from_pretrained(
        'Qwen/Qwen2.5-0.5B-Instruct',
        torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32,
        attn_implementation="sdpa"
    ).eval().to(device)

    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')
    tokenizer.pad_token = tokenizer.eos_token
except Exception as e:
    logger.error(f"Ошибка загрузки модели: {e}")
    raise

EOS_TOKEN_ID = 151645
MAX_LENGTH = 1000

input_text_hedgehog = '''<|im_start|>system
You are a storyteller. Generate a story based on user message.<|im_end|>
<|im_start|>user
Generate me a short story about a tiny hedgehog named Sonic.<|im_end|>
<|im_start|>assistant
'''

input_text_json = '''<|im_start|>system
You are a JSON machine. Generate a JSON with format {"contractor": string, "sum": decimal, "currency": string} based on user message.<|im_end|>
<|im_start|>user
Transfer 100 rubles and 50 kopeck to Mike<|im_end|>
<|im_start|>assistant
'''