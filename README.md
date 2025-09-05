# Medical RAG Chatbot 🏥🤖

A Retrieval-Augmented Generation (RAG) chatbot specifically designed for answering medical questions using PDF documents as knowledge base. Built with LangChain, HuggingFace, and FAISS.

## 🚀 Features

- **PDF Processing**: Automatically loads and processes medical PDF documents
- **Intelligent Chunking**: Splits documents into optimal chunks for retrieval
- **Vector Search**: Uses FAISS for fast and accurate document retrieval
- **Medical-Focused**: Custom prompt template optimized for medical Q&A
- **HuggingFace Integration**: Leverages Mistral-7B-Instruct model for responses
- **Modular Architecture**: Clean, maintainable codebase with proper error handling

## 🏗️ Architecture

```
app/
├── components/           # Core RAG pipeline components
│   ├── pdf_loader.py    # Document loading and chunking
│   ├── embeddings.py    # HuggingFace embeddings
│   ├── vector_store.py  # FAISS vector database
│   ├── llm.py          # Language model integration
│   ├── retriever.py    # QA chain with custom prompts
│   └── data_loader.py  # Main processing pipeline
├── config/
│   └── config.py       # Configuration management
├── common/
│   ├── logger.py       # Logging utilities
│   └── custom_exception.py # Exception handling
└── templates/          # Future UI templates
```

## 🛠️ Installation

1. **Clone the repository**
```bash
git clone https://github.com/0Xuser100/medical-rag-chatbot.git
cd medical-rag-chatbot
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
```bash
cp app/.env.example app/.env
```
Edit `app/.env` and add your HuggingFace token:
```
HF_TOKEN=your_huggingface_token_here
HUGGINGFACE_API_TOKEN=your_huggingface_token_here
```

## 📚 Usage

### 1. Prepare Your Data
- Place your medical PDF files in the `data/` directory
- Supported format: PDF files only

### 2. Process Documents and Create Vector Store
```bash
# Run from project root
python -m app.components.data_loader

# Or alternatively
python app/components/data_loader.py
```

### 3. Query the System
```python
from app.components.retriever import create_qa_chain

# Initialize the QA chain
qa_chain = create_qa_chain()

# Ask medical questions
response = qa_chain.invoke({"query": "What are the symptoms of diabetes?"})
print(response["result"])
```

## 🔧 Configuration

Key configuration parameters in `app/config/config.py`:

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `HUGGINGFACE_REPO_ID` | `mistralai/Mistral-7B-Instruct-v0.3` | LLM model |
| `DB_FAISS_PATH` | `vectorstore/db_faiss` | Vector store location |
| `DATA_PATH` | `data/` | PDF documents directory |
| `CHUNK_SIZE` | `800` | Text chunk size |
| `CHUNK_OVERLAP` | `100` | Overlap between chunks |

## 🔍 How It Works

1. **Document Loading**: PDFs are loaded from the `data/` directory using LangChain's PyPDFLoader
2. **Text Chunking**: Documents are split into overlapping chunks for optimal retrieval
3. **Embeddings**: Text chunks are converted to embeddings using `sentence-transformers/all-MiniLM-L6-v2`
4. **Vector Storage**: Embeddings are stored in FAISS for fast similarity search
5. **Query Processing**: User questions are embedded and matched against the knowledge base
6. **Response Generation**: Retrieved context is sent to Mistral-7B model with a medical-focused prompt

## 📋 Requirements

- Python 3.8+
- HuggingFace account and API token
- At least 4GB RAM (for model inference)
- GPU recommended but not required

## 🚧 Current Limitations

- Supports PDF files only
- Responses limited to 2-3 lines (configurable in prompt)
- Requires HuggingFace token for model access
- English language only

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [LangChain](https://langchain.com/) for the RAG framework
- [HuggingFace](https://huggingface.co/) for models and embeddings
- [FAISS](https://github.com/facebookresearch/faiss) for vector similarity search
- [Mistral AI](https://mistral.ai/) for the language model

## 📞 Support

If you have questions or need help, please:
- Open an issue on GitHub
- Check the existing issues for similar problems
- Review the documentation in `CLAUDE.md`

---

**⚠️ Medical Disclaimer**: This chatbot is for informational purposes only and should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare providers for medical decisions.