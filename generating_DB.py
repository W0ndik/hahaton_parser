import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
import os

# Инициализация клиента ChromaDB
client = chromadb.Client()

# Создание или подключение к коллекции для текста и изображений
text_collection = client.create_collection("text_embeddings")
image_collection = client.create_collection("image_embeddings")

# Модели для генерации эмбеддингов
text_model = SentenceTransformer('all-MiniLM-L6-v2')  # Вы можете использовать свою модель
clip_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch16')
clip_processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch16')

# Функция для сохранения эмбеддингов текста в ChromaDB
def save_text_embeddings(text_data, embeddings, collection):
    for idx, (text, embedding) in enumerate(zip(text_data, embeddings)):
        collection.add(
            ids=[f"text_{idx}"],  # Идентификатор для текста
            embeddings=[embedding],  # Эмбеддинг текста
            metadatas=[{"text": text}],  # Метаданные текста
            documents=[text]  # Сами тексты
        )
    print(f"Добавлено {len(text_data)} текстов в ChromaDB.")

# Функция для сохранения эмбеддингов изображений в ChromaDB
def save_image_embeddings(image_data, embeddings, collection):
    for idx, (image, embedding) in enumerate(zip(image_data, embeddings)):
        collection.add(
            ids=[f"image_{idx}"],  # Идентификатор для изображения
            embeddings=[embedding],  # Эмбеддинг изображения
            metadatas=[{"image": image}],  # Метаданные изображения
            documents=[image]  # Путь к изображению
        )
    print(f"Добавлено {len(image_data)} изображений в ChromaDB.")

# Функция для генерации эмбеддингов текста
def generate_text_embeddings(text_data):
    embeddings = text_model.encode(text_data)
    return embeddings

# Функция для генерации эмбеддингов изображений
def generate_image_embeddings(image_paths):
    embeddings = []
    for image_path in image_paths:
        image = Image.open(image_path)
        inputs = clip_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            image_embedding = clip_model.get_image_features(**inputs)
        embeddings.append(image_embedding.cpu().numpy())
    
    return np.array(embeddings)

# Функция для обработки текстовых данных и сохранения эмбеддингов
def process_text_data(text_data):
    print("Генерация эмбеддингов для текста...")
    text_embeddings = generate_text_embeddings(text_data)
    save_text_embeddings(text_data, text_embeddings, text_collection)

# Функция для обработки изображений и сохранения эмбеддингов
def process_image_data(image_paths):
    print("Генерация эмбеддингов для изображений...")
    image_embeddings = generate_image_embeddings(image_paths)
    save_image_embeddings(image_paths, image_embeddings, image_collection)

# Пример использования
if __name__ == "__main__":
    # Пример текстовых данных для парсинга
    text_data = [
        "Как настроить весы распределителя Amazone.",
        "Инструкция по эксплуатации Amazone.",
        "Обзор функций и настроек.",
        "Технические характеристики модели."
    ]
    
    # Пример путей к изображениям
    image_paths = [
        "images/page_1.png",  # Путь к изображению
        "images/page_2.png",  # Путь к изображению
        "images/page_3.png"   # Путь к изображению
    ]

    # Обработка и сохранение текстовых эмбеддингов в ChromaDB
    process_text_data(text_data)

    # Обработка и сохранение эмбеддингов изображений в ChromaDB
    process_image_data(image_paths)
