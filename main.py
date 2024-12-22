from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import uvicorn
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse



# Инициализация приложения FastAPI
app = FastAPI(
    title="LLaMA 3-Instruct API",
    description="API для инференса LLaMA 3-Instruct",
    version="1.0.0",
)

# Подключение статики
app.mount("/static", StaticFiles(directory="static"), name="static")

# Инициализация модели и токенайзера
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"  # Укажите имя модели из Hugging Face
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

quantization_config = BitsAndBytesConfig(load_in_4bit=True,
                                         bnb_4bit_compute_dtype=torch.float16)

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME,
                                                 quantization_config=quantization_config,
                                                 device_map="auto",
                                                 torch_dtype=torch.float16)
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



def get_prompt(text):
    prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2024\nToday Date: 28 Dec 2024\n\nYou are assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"""
    return prompt.format(message=text)


# Корневой маршрут
@app.get("/")
def redirect_to_ui():
    return RedirectResponse(url="/static/index.html")


@app.post("/generate", response_model=InferenceResponse)
def generate_text(request: InferenceRequest):
    try:
        # Токенизация ввода
        inputs = tokenizer.encode(request.prompt, return_tensors="pt").to(device)

        # Генерация текста
        outputs = model.generate(
            inputs,
            max_length=request.max_length,
            temperature=request.temperature,
            top_k=request.top_k,
            do_sample=True,
        )

        # Декодирование результата
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return InferenceResponse(generated_text=generated_text)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# # Запуск: uvicorn main:app --reload
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
