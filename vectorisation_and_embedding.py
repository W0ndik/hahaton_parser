import os
import json
import torch
import base64
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from io import BytesIO

# Загрузка моделей
def load_text_model():
    """Загружает модель для векторизации текста."""
    print("Загрузка модели Sentence Transformers для текста...")
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')  # Вы можете использовать свою модель
    print("Модель для текста загружена.")
    return model

def load_clip_model():
    """Загружает модель CLIP для векторизации изображений."""
    print("Загрузка модели CLIP для изображений...")
    model = CLIPModel.from_pretrained('openai/clip-vit-base-patch16')
    processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch16')
    print("Модель CLIP загружена.")
    return model, processor

def vectorize_text(model, text_data):
    """Создание векторных эмбеддингов для текста."""
    print("Векторизация текста...")
    text_embeddings = model.encode(text_data, convert_to_tensor=True)
    return text_embeddings

def vectorize_images(model, processor, image_data):
    """Создание векторных эмбеддингов для изображений."""
    print("Векторизация изображений...")
    image_embeddings = []
    for img_data in image_data:
        # Декодируем изображение из base64
        img_bytes = base64.b64decode(img_data.split(",")[1])
        img = Image.open(BytesIO(img_bytes))
        
        # Преобразуем изображение для CLIP
        inputs = processor(images=img, return_tensors="pt", padding=True)
        
        with torch.no_grad():
            # Получаем эмбеддинг изображения
            image_emb = model.get_image_features(**inputs)
        
        image_embeddings.append(image_emb)
    return image_embeddings

def process_json_and_generate_embeddings(json_path, output_folder="embeddings_output"):
    """Загружает JSON, создает эмбеддинги и сохраняет их."""
    # Создаем выходную папку для эмбеддингов
    os.makedirs(output_folder, exist_ok=True)

    # Загружаем данные из JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Разделяем данные на текст и изображения
    text_data = [item['content'] for item in data if item['type'] == 'text']
    image_data = [item['content'] for item in data if item['type'] == 'image']

    # Загружаем модели
    text_model = load_text_model()
    clip_model, processor = load_clip_model()

    # Векторизация текста и изображений
    text_embeddings = vectorize_text(text_model, text_data)
    image_embeddings = vectorize_images(clip_model, processor, image_data)

    # Сохраняем эмбеддинги в файлы
    text_embeddings_file = os.path.join(output_folder, f"{os.path.basename(json_path)}_text_embeddings.npy")
    image_embeddings_file = os.path.join(output_folder, f"{os.path.basename(json_path)}_image_embeddings.npy")
    
    # Сохраняем векторные эмбеддинги в файлы
    torch.save(text_embeddings, text_embeddings_file)
    torch.save(image_embeddings, image_embeddings_file)

    print(f"Эмбеддинги для файла {json_path} сохранены в {output_folder}")

def process_all_json_files_in_folder(input_folder="parsed_output", output_folder="embeddings_output"):
    """Обрабатывает все JSON файлы в папке и генерирует для них эмбеддинги."""
    if not os.path.exists(input_folder):
        print(f"Папка {input_folder} не существует.")
        return

    # Получаем все JSON файлы в папке
    json_files = [f for f in os.listdir(input_folder) if f.endswith(".json")]
    total_files = len(json_files)

    if total_files == 0:
        print(f"В папке {input_folder} нет JSON файлов.")
        return

    print(f"Всего JSON файлов для обработки: {total_files}")

    for file_name in json_files:
        json_path = os.path.join(input_folder, file_name)
        process_json_and_generate_embeddings(json_path, output_folder)

if __name__ == "__main__":
    # Обрабатываем все JSON файлы в папке parsed_output
    input_folder = "parsed_output"
    output_folder = "embeddings_output"
    process_all_json_files_in_folder(input_folder, output_folder)
