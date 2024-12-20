import os
from llama_index.core import Document, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.embeddings import BaseEmbedding
import numpy as np


class CustomEmbedding(BaseEmbedding):
    def __init__(self):
        super().__init__()

    def embed(self, text: str) -> list[float]:
        """Генерация вектора для текста."""
        return list(np.random.rand(300))

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Обработка списка текстов."""
        return [self.embed(text) for text in texts]

    def _get_text_embedding(self, text: str) -> list[float]:
        """Метод для получения текстового вектора."""
        return self.embed(text)

    def _get_query_embedding(self, query: str) -> list[float]:
        """Метод для получения вектора запроса."""
        return self.embed(query)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        """Асинхронный метод для получения вектора запроса."""
        return self.embed(query)


def index_files_from_directory(directory_path):
    """Индексирует все текстовые файлы из указанной папки."""
    if not os.path.exists(directory_path):
        print(f"Папка {directory_path} не существует.")
        return None

    all_files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith('.json')]
    if len(all_files) == 0:
        print(f"В папке {directory_path} нет файлов для индексации.")
        return None

    documents = []
    for file_path in all_files:
        print(f"Индексируем файл: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            documents.append(Document(content=content))

    # Используем кастомный embedding
    custom_embed_model = CustomEmbedding()
    index = VectorStoreIndex.from_documents(documents, embed_model=custom_embed_model)
    print(f"Индексация завершена. Всего документов: {len(documents)}.")
    return index


def save_index(index, index_path):
    """Сохраняет индекс на диск."""
    if index:
        index.storage_context.persist(persist_dir=index_path)
        print(f"Индекс сохранен в папку {index_path}.")
    else:
        print("Нет индекса для сохранения.")


def load_index(index_path):
    """Загружает индекс из указанной папки."""
    storage_context = StorageContext.from_defaults(persist_dir=index_path)
    index = load_index_from_storage(storage_context)
    print(f"Индекс успешно загружен из папки {index_path}.")
    return index


if __name__ == "__main__":
    folder_path = "parsed_output"
    index = index_files_from_directory(folder_path)
    save_index(index, 'index_storage')  # Указываем папку для сохранения индекса
