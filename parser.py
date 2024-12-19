import pdfplumber
import base64
import os
import json
from pdf2image import convert_from_path
from PIL import Image
from tqdm import tqdm  # Используем tqdm для красивого прогресса

# Задаём poppler_path
poppler_path = os.getenv("POPPLER_PATH", r"C:\Users\sgs-w\OneDrive\Рабочий стол\ХАХАТОН\poppler-24.08.0\Library\bin")

# Функция для извлечения текста из PDF
def extract_text_from_pdf(pdf_path):
    text_data = []
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        for page_number, page in enumerate(tqdm(pdf.pages, desc="Извлечение текста", unit="страниц"), start=1):
            text = page.extract_text()
            if text:
                text_data.append({
                    "page": page_number,
                    "content": text,
                    "type": "text"
                })
    return text_data

# Функция для извлечения таблиц из PDF
def extract_tables_from_pdf(pdf_path):
    tables_data = []
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        for page_number, page in enumerate(tqdm(pdf.pages, desc="Извлечение таблиц", unit="страниц"), start=1):
            tables = page.extract_tables()
            for table in tables:
                table_data = []
                for row in table[1:]:
                    if len(row) == len(table[0]):  # Проверяем длину строки
                        table_data.append(dict(zip(table[0], row)))
                tables_data.append({
                    "page": page_number,
                    "content": table_data,
                    "type": "table"
                })
    return tables_data

# Функция для извлечения изображений из PDF
def extract_images_from_pdf(pdf_path, output_folder="images", poppler_path=None):
    os.makedirs(output_folder, exist_ok=True)

    images_data = []
    images = convert_from_path(pdf_path, poppler_path=poppler_path)
    total_pages = len(images)

    for page_number, image in enumerate(tqdm(images, desc="Извлечение изображений", unit="страниц"), start=1):
        image_path = os.path.join(output_folder, f"page_{page_number}.png")
        image.save(image_path, "PNG")

        with open(image_path, "rb") as img_file:
            encoded_string = base64.b64encode(img_file.read()).decode('utf-8')
            images_data.append({
                "page": page_number,
                "content": f"data:image/png;base64,{encoded_string}",
                "type": "image"
            })

    return images_data

# Функция для объединения текста, таблиц и изображений в JSON
def parse_pdf_to_json(pdf_path, output_json_path, output_image_folder="images"):
    try:
        print(f"Обработка файла: {os.path.basename(pdf_path)}")
        text_data = extract_text_from_pdf(pdf_path)
        tables_data = extract_tables_from_pdf(pdf_path)
        images_data = extract_images_from_pdf(pdf_path, output_folder=output_image_folder, poppler_path=poppler_path)

        combined_data = text_data + tables_data + images_data
        combined_data.sort(key=lambda x: x.get("page", float('inf')))

        with open(output_json_path, "w", encoding="utf-8") as json_file:
            json.dump(combined_data, json_file, ensure_ascii=False, indent=4)

        print(f"Данные сохранены в {output_json_path}")
    except Exception as e:
        print(f"Ошибка при обработке {pdf_path}: {e}")

# Функция для обработки всех PDF в папке
def parse_all_pdfs_in_folder(folder_path, output_folder="output", image_folder="images"):
    if not os.path.exists(folder_path):
        print(f"Папка {folder_path} не существует.")
        return

    os.makedirs(output_folder, exist_ok=True)

    pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]
    total_files = len(pdf_files)

    if total_files == 0:
        print("В папке нет PDF-файлов для обработки.")
        return

    print(f"Всего PDF-файлов для обработки: {total_files}")

    for file_number, file_name in enumerate(tqdm(pdf_files, desc="Обработка PDF файлов", unit="файлов"), start=1):
        pdf_path = os.path.join(folder_path, file_name)
        output_json_path = os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}.json")
        parse_pdf_to_json(pdf_path, output_json_path, image_folder)

# Пример использования
if __name__ == "__main__":
    folder_path = "parse"
    output_folder = "parsed_output"
    image_folder = "images"

    parse_all_pdfs_in_folder(folder_path, output_folder, image_folder)
