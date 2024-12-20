# chroma stuff
import chromadb
from chromadb.config import Settings

# ja ne narkoman
import torch

# loading embeddings
import numpy as np

# file listing
import glob
import os

chroma_client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory="db/"))

collection = chroma_client.get_or_create_collection(name="amazone")

json_files = []
for file in glob.glob('../parsed_output/*.json'):
    json_files.append(file)

for file in json_files:
    with open(file) as f:
        print('Working on file ' + file)
        contents = f.read()

        # emgeddings are being created by model all-MiniLM-L6-v2 automatically by chromadb
        collection.upsert(
            documents = [
                contents
            ],
            metadatas = [
                {'doc': file}
            ],
            ids = [
                file
            ]
        )
