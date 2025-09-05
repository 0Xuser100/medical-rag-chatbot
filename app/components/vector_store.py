from langchain_community.vectorstores import FAISS
import os
from components.embeddings import get_embedding_model

from common.logger import get_logger
from common.custom_exception import CustomException

from config.config import DB_FAISS_PATH

logger = get_logger(__name__)

def load_vector_store():
    try:
        embedding_model = get_embedding_model()

        if os.path.exists(DB_FAISS_PATH):
            logger.info("Loading existing vectorstore...")
            
            # Check if both index files exist
            index_path = os.path.join(DB_FAISS_PATH, "index.faiss")
            pkl_path = os.path.join(DB_FAISS_PATH, "index.pkl")
            
            if not os.path.exists(index_path) or not os.path.exists(pkl_path):
                error_message = CustomException(f"Vector store files missing. Expected: {index_path} and {pkl_path}")
                logger.error(str(error_message))
                raise error_message
            
            try:
                return FAISS.load_local(
                    DB_FAISS_PATH,
                    embedding_model,
                    allow_dangerous_deserialization=True
                )
            except Exception as load_error:
                logger.warning(f"Failed to load existing vector store: {load_error}")
                logger.info("Vector store may be incompatible with current embedding model")
                # Try to recreate the vector store
                return recreate_vector_store()
        else:
            error_message = CustomException("No vector store found at path: " + DB_FAISS_PATH)
            logger.error(str(error_message))
            raise error_message

    except Exception as e:
        error_message = CustomException("Failed to load vectorstore" , e)
        logger.error(str(error_message))
        raise error_message

def recreate_vector_store():
    """Recreate vector store from PDF data if it's incompatible"""
    try:
        from components.pdf_loader import load_pdf_files, create_text_chunks
        from config.config import DATA_PATH
        
        logger.info("Recreating vector store from source data...")
        
        # Check if data directory exists
        if not os.path.exists(DATA_PATH):
            raise CustomException(f"Data directory not found: {DATA_PATH}")
        
        # Load PDF documents
        documents = load_pdf_files()
        if not documents:
            raise CustomException("No documents loaded from PDFs")
        
        # Create text chunks
        text_chunks = create_text_chunks(documents)
        if not text_chunks:
            raise CustomException("No text chunks extracted from PDFs")
        
        # Remove old vector store
        import shutil
        if os.path.exists(DB_FAISS_PATH):
            shutil.rmtree(DB_FAISS_PATH)
        
        # Create new vector store
        return save_vector_store(text_chunks)
        
    except Exception as e:
        logger.error(f"Failed to recreate vector store: {e}")
        raise CustomException("Could not recreate vector store", e)

# Creating new vectorstore function
def save_vector_store(text_chunks):
    try:
        if not text_chunks:
            raise CustomException("No chunks were found..")
        
        logger.info("Generating your new vectorstore")

        embedding_model = get_embedding_model()

        db = FAISS.from_documents(text_chunks,embedding_model)

        logger.info("Saving vectorstoree")

        db.save_local(DB_FAISS_PATH)

        logger.info("Vectostore saved sucesfulyy...")

        return db
    
    except Exception as e:
        error_message = CustomException("Failed to create new vectorstore " , e)
        logger.error(str(error_message))
        raise error_message
    

