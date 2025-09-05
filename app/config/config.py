import os
from dotenv import load_dotenv
load_dotenv()
OPEN_AI_API_KEY = os.environ.get("OPENAI_API_KEY")

OPEN_AI_MODEL="gpt-5-mini"
DB_FAISS_PATH="vectorstore/db_faiss"
DATA_PATH="data/"
CHUNK_SIZE=500
CHUNK_OVERLAP=50
