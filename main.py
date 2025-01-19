from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="LLaMA 3-Instruct API",
    description="API для инференса LLaMA 3-Instruct",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Или укажите конкретные адреса, например, ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],  # Разрешить все методы: GET, POST, OPTIONS, PUT, DELETE и т.д.
    allow_headers=["*"],  # Разрешить все заголовки
)

# Инициализация модели и токенайзера
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"  # Укажите имя модели из Hugging Face
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    if device == "cuda":
        print('cuda')
        from transformers import BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(load_in_4bit=True,
                                                 bnb_4bit_compute_dtype=torch.float16)

        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME,
                                                     quantization_config=quantization_config,
                                                     device_map="auto",
                                                     torch_dtype=torch.float16)
    else:
        print('cpu')
        model = None
        # model = AutoModelForCausalLM.from_pretrained(MODEL_NAME,
        #                                              device_map='cpu',
        #                                              low_cpu_mem_usage=True)
        #
        # model = torch.quantization.quantize_dynamic(
        #     model, {torch.nn.Linear}, dtype=torch.qint8
        # )

except Exception as e:
    raise RuntimeError(f"Ошибка загрузки модели: {e}")


# Схема данных для запросов
class InferenceRequest(BaseModel):
    prompt: str
    max_length: int = 128  # Максимальная длина текста
    temperature: float = 0.7  # Температура для управления разнообразием
    top_k: int = 50  # Используется для сужения набора токенов на основе вероятности


# Схема данных для ответов
class InferenceResponse(BaseModel):
    generated_text: str
    answer: str


def get_last_answer(text):
    return text.split('<|end_header_id|>\n\n')[-1].replace('<|eot_id|>', '')


def make_prompt_chat(chat):
    prompt = ["<|begin_of_text|>"]
    for message in chat:
        role = message['role']
        content = message['content']
        prompt.append(f"<|start_header_id|>{role}<|end_header_id|>\n\n")
        if role == 'system':
            prompt.append('Cutting Knowledge Date: December 2024\nToday Date: 28 Dec 2024\n\n')
        if content:
            prompt.append(content)
            prompt.append(f"<|eot_id|>")
    return ''.join(prompt)


def get_prompt(text):
    prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2024\nToday Date: 28 Dec 2024\n\nYou are assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"""
    return prompt.format(message=text)


d = {'counter': 0}


@app.post("/generate", response_model=InferenceResponse)
def generate_text(request: InferenceRequest):
    try:

        if device == "cuda":
            # Токенизация ввода
            inputs = tokenizer(get_prompt(request.prompt), return_tensors="pt", padding=True, truncation=True).to(
                device)

            # Установка attention_mask
            attention_mask = inputs["attention_mask"]
            # Генерация текста
            outputs = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=request.max_length,
                temperature=request.temperature,
                top_k=request.top_k,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )

            # Декодирование результата
            generated_text = tokenizer.decode(outputs[0])
        else:
            d['counter'] += 1
            generated_text = get_prompt(request.prompt) + f'Ответ {d["counter"]} от модели cpu' + '<|eot_id|>'
        answer = get_last_answer(generated_text)
        return InferenceResponse(generated_text=generated_text, answer=answer)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
