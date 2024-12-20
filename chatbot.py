from chromadb import Client
from chromadb.config import Settings

# Настройки
CHROMA_DIR = "chroma_db"
PROMPT = """
Выберите действие:
1. Ввести запрос
2. Выйти
Ваш выбор: """

def init_chroma():
    """Инициализация ChromaDB."""
    return Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=CHROMA_DIR))

def query_chroma(db_client, query_text):
    """Выполнить запрос к базе ChromaDB."""
    collection = db_client.get_or_create_collection("ammonia_tech_docs")
    results = collection.query(query_texts=[query_text], n_results=3)
    return results

def run_prompt():
    """Запуск простого интерфейса для тестирования."""
    client = init_chroma()
    print("ChromaDB инициализирована. База готова для запросов.")

    while True:
        choice = input(PROMPT).strip()
        
        if choice == "1":
            query = input("Введите ваш запрос: ").strip()
            if not query:
                print("Запрос пустой. Попробуйте снова.")
                continue
            
            results = query_chroma(client, query)
            if results and results.get("documents"):
                print("\nРезультаты:")
                for i, doc in enumerate(results["documents"], start=1):
                    print(f"{i}. {doc}\n")
            else:
                print("Ничего не найдено по вашему запросу.\n")
        
        elif choice == "2":
            print("Выход.")
            break
        
        else:
            print("Некорректный выбор. Попробуйте снова.\n")

if __name__ == "__main__":
    run_prompt()
