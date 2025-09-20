import os
from dotenv import load_dotenv

load_dotenv()
OPEN_AI_API_KEY = os.getenv("OPENAI_API_KEY")
public_key = os.getenv("public_key")
secret_key = os.getenv("secret_key")
host = os.environ.get("host")


OPEN_AI_MODEL = "gpt-4o-mini"
DB_FAISS_PATH = "vectorstore/db_faiss"
DATA_PATH = "data/"
CHUNK_SIZE = 900
CHUNK_OVERLAP = 90

# Memory configuration
MEMORY_SUMMARY_THRESHOLD = 8  # switch to summary memory after this many messages
MEMORY_WINDOW_TOKEN_LIMIT = 800  # token budget for window memory
MEMORY_WINDOW_DEFAULT_K = 2  # default number of recent exchanges to keep
MEMORY_WINDOW_MAX_K = 3  # cap on exchanges kept when window memory is active
MEMORY_SUMMARY_TOKEN_LIMIT = 1200  # token limit for summary memory compression
MEMORY_SUMMARY_RECENT_K = (
    4  # number of recent exchanges to retain alongside the summary
)
RETRIEVER_TOP_K = int(
    os.getenv("RETRIEVER_TOP_K", 5)
)  # documents retrieved per query (override via env)
