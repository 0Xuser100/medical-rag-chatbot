import os
from dotenv import load_dotenv
load_dotenv()
OPEN_AI_API_KEY = os.getenv("OPENAI_API_KEY")
public_key=os.getenv("public_key")
secret_key=os.getenv("secret_key")
host=os.environ.get("host")


OPEN_AI_MODEL="gpt-4o-mini"
DB_FAISS_PATH="vectorstore/db_faiss"
DATA_PATH="data/"
CHUNK_SIZE=1000
CHUNK_OVERLAP=120
