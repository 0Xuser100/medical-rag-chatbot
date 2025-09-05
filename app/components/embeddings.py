from langchain_openai import OpenAIEmbeddings


from common.logger import get_logger
from common.custom_exception import CustomException

logger = get_logger(__name__)

def get_embedding_model():
    try:
        logger.info("Initializing OpenAI embedding model")

        model = OpenAIEmbeddings( model="text-embedding-3-small",dimensions=1536)

        logger.info("OpenAI embedding model loaded successfully....")

        return model
    
    except Exception as e:
        error_message=CustomException("Error occured while loading embedding model" , e)
        logger.error(str(error_message))
        raise error_message