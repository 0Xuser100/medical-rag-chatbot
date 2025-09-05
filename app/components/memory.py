from langchain.memory import ConversationBufferWindowMemory, ConversationSummaryBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from components.llm import load_llm
from components.vector_store import load_vector_store
from config.config import OPEN_AI_MODEL, OPEN_AI_API_KEY
from common.logger import get_logger
from common.custom_exception import CustomException

logger = get_logger(__name__)

CONVERSATIONAL_PROMPT_TEMPLATE = """You are a helpful medical AI assistant. Use the following context and conversation history to answer the user's question accurately and helpfully.

Context from medical documents:
{context}

Previous conversation:
{chat_history}

Current question: {question}

Instructions:
- Provide clear, accurate medical information based on the context
- Reference previous conversation when relevant
- If the question relates to earlier discussion, acknowledge that connection
- Keep responses informative but concise (2-4 sentences typically)
- If information isn't available in the context, say so clearly

Answer:"""

class ConversationalMemory:
    def __init__(self, memory_type="window", max_token_limit=1000, k=3):
        """
        Initialize conversational memory
        
        Args:
            memory_type: "window" for recent messages, "summary" for summarized long conversations
            max_token_limit: Maximum tokens to keep in memory
            k: Number of previous messages to keep (for window memory)
        """
        self.memory_type = memory_type
        self.max_token_limit = max_token_limit
        self.k = k
        self.llm = None
        self.memory = None
        
    def _get_llm(self):
        """Get LLM instance (cached)"""
        if self.llm is None:
            self.llm = load_llm(OPEN_AI_MODEL, OPEN_AI_API_KEY)
        return self.llm
    
    def create_memory(self):
        """Create appropriate memory type"""
        try:
            if self.memory_type == "window":
                # Keep last k conversation turns
                self.memory = ConversationBufferWindowMemory(
                    k=self.k,
                    memory_key="chat_history",
                    return_messages=True,
                    output_key="answer"
                )
            elif self.memory_type == "summary":
                # Summarize old conversations, keep recent ones
                llm = self._get_llm()
                self.memory = ConversationSummaryBufferMemory(
                    llm=llm,
                    max_token_limit=self.max_token_limit,
                    memory_key="chat_history",
                    return_messages=True,
                    output_key="answer"
                )
            else:
                raise CustomException(f"Unknown memory type: {self.memory_type}")
                
            logger.info(f"Created {self.memory_type} memory with limit {self.max_token_limit}")
            return self.memory
            
        except Exception as e:
            logger.error(f"Failed to create memory: {e}")
            raise CustomException("Failed to create conversational memory", e)
    
    def get_memory(self):
        """Get or create memory instance"""
        if self.memory is None:
            self.create_memory()
        return self.memory

def create_conversational_qa_chain(memory_type="window", max_token_limit=1000, k=3):
    """
    Create a conversational QA chain with memory
    
    Args:
        memory_type: "window" for recent messages, "summary" for summarized conversations
        max_token_limit: Maximum tokens in memory
        k: Number of recent conversation turns to keep
    """
    try:
        logger.info("Creating conversational QA chain with memory")
        
        # Load components
        db = load_vector_store()
        if db is None:
            raise CustomException("Vector store not present or empty")
            
        llm = load_llm(OPEN_AI_MODEL, OPEN_AI_API_KEY)
        if llm is None:
            raise CustomException("LLM not loaded")
        
        # Create memory
        memory_manager = ConversationalMemory(memory_type, max_token_limit, k)
        memory = memory_manager.get_memory()
        
        # Create custom prompt
        custom_prompt = PromptTemplate(
            template=CONVERSATIONAL_PROMPT_TEMPLATE,
            input_variables=["context", "chat_history", "question"]
        )
        
        # Create conversational retrieval chain
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=db.as_retriever(search_kwargs={'k': 5}),
            memory=memory,
            combine_docs_chain_kwargs={'prompt': custom_prompt},
            verbose=False,
            return_source_documents=False
        )
        
        logger.info("Successfully created conversational QA chain")
        return qa_chain
        
    except Exception as e:
        error_message = CustomException("Failed to create conversational QA chain", e)
        logger.error(str(error_message))
        raise error_message

def create_session_qa_chain(session_messages=None, memory_type="window"):
    """
    Create QA chain and populate with existing session messages
    
    Args:
        session_messages: List of message dicts with 'role' and 'content'
        memory_type: Type of memory to use
    """
    try:
        # Determine memory settings based on conversation length
        if session_messages:
            msg_count = len(session_messages)
            if msg_count > 10:
                # Use summary memory for long conversations
                memory_type = "summary"
                max_tokens = 1500
                k = 6
            else:
                # Use window memory for short conversations
                memory_type = "window" 
                max_tokens = 1000
                k = min(msg_count, 4)
        else:
            memory_type = "window"
            max_tokens = 1000
            k = 3
        
        # Create QA chain
        qa_chain = create_conversational_qa_chain(memory_type, max_tokens, k)
        
        # Populate memory with existing conversation
        if session_messages and len(session_messages) > 1:
            logger.info(f"Populating memory with {len(session_messages)} existing messages")
            
            for i in range(0, len(session_messages), 2):
                if i + 1 < len(session_messages):
                    user_msg = session_messages[i]
                    assistant_msg = session_messages[i + 1]
                    
                    if user_msg['role'] == 'user' and assistant_msg['role'] == 'assistant':
                        # Add to memory
                        qa_chain.memory.chat_memory.add_user_message(user_msg['content'])
                        qa_chain.memory.chat_memory.add_ai_message(assistant_msg['content'])
        
        return qa_chain
        
    except Exception as e:
        logger.error(f"Failed to create session QA chain: {e}")
        raise CustomException("Failed to create session QA chain", e)