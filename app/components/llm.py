from langchain_openai import ChatOpenAI
from config.config import OPEN_AI_API_KEY,OPEN_AI_MODEL

from common.logger import get_logger
from common.custom_exception import CustomException

logger = get_logger(__name__)

def load_llm(OPEN_AI_MODEL: str = OPEN_AI_MODEL ,OPEN_AI_API_KEY :str = OPEN_AI_API_KEY):
    try:
        logger.info("Loading OpenAI LLM")

        llm = ChatOpenAI(
                model=OPEN_AI_MODEL,
                api_key=OPEN_AI_API_KEY,
            )

        logger.info("OpenAI LLM loaded successfully...")

        return llm
    
    except Exception as e:
        error_message = CustomException("Failed to load a llm" , e)
        logger.error(str(error_message))
        raise error_message