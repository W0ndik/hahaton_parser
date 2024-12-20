from transformers import AutoTokenizer, AutoModelForCausalLM

# Модель и токенизатор
MODEL_NAME = "sberbank-ai/rugpt3small_based_on_gpt2"

def load_rugpt3():
    """Загружает RuGPT3."""
    print("Загрузка модели RuGPT3...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    print("Модель RuGPT3 загружена.")
    return tokenizer, model

def generate_text(prompt, tokenizer, model, max_length=200):
    """Генерирует текст на основе запроса."""
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs["input_ids"], max_length=max_length, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    # Загрузка модели
    tokenizer, model = load_rugpt3()
    
    # Пример использования
    prompt = input("Введите запрос для генерации текста: ")
    result = generate_text(prompt, tokenizer, model)
    print("\nСгенерированный текст:\n", result)
