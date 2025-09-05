# 🏥 Medical RAG Chatbot - Version 1.0

An intelligent medical question-answering chatbot built with **OpenAI GPT**, **LangChain**, and **FAISS** vector search. Get accurate, contextual answers to medical questions based on medical literature with advanced conversational memory.

## ✨ Features

- 🤖 **OpenAI GPT Integration** - Powered by GPT-4o-mini for accurate responses
- 🧠 **Advanced Memory System** - Remembers conversation context with smart memory management
- 📚 **Medical Knowledge Base** - FAISS vector search through medical documents
- 💬 **Modern Chat Interface** - Beautiful, responsive web UI with real-time interactions
- 📱 **Mobile Responsive** - Works seamlessly on all devices
- 💾 **Export Conversations** - Download chat history as text files
- 🔄 **Smart Memory Types** - Window memory for short chats, summary memory for long conversations
- 🎯 **Context Awareness** - AI references previous questions and builds on earlier answers

## 🚀 Quick Start

### Prerequisites

- Python 3.12
- **Conda** package manager
- OpenAI API key (get from [OpenAI Platform](https://platform.openai.com/))

### 1. Environment Setup with Conda

```bash
# Clone the repository
git clone https://github.com/0Xuser100/medical-rag-chatbot.git
cd MedicalRag

# Create conda environment with Python 3.12
conda create -n medical-rag python=3.12 -y
conda activate medical-rag

# Navigate to app directory and install dependencies
cd app
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your OpenAI API key
OPENAI_API_KEY="your_openai_api_key_here"
```

### 3. Prepare Medical Knowledge Base

```bash
# Place your medical PDF files in the data/ directory
# Example: data/medical_encyclopedia.pdf

# Create FAISS vector store from PDFs
python components/data_loader.py
```

### 4. Launch the Chatbot

```bash
# Start the Flask application
python application.py
```

🎉 **Visit `http://localhost:5000` to start chatting with your medical AI assistant!**

## 🧠 Advanced Memory System

### How Memory Works

The chatbot features sophisticated conversational memory:

- **🔄 Context Retention**: Remembers what you've discussed throughout the conversation
- **🎯 Follow-up Awareness**: Understands references like "What are the symptoms?" after asking about diabetes
- **📊 Smart Memory Types**:
  - **Window Memory** (≤10 messages): Keeps recent conversation turns
  - **Summary Memory** (>10 messages): Summarizes old content, maintains recent context
- **💭 Memory Indicators**: UI shows conversation length and active memory type

### Example Conversation Flow

```
You: "What is diabetes?"
AI: "Diabetes is a chronic condition where blood sugar levels are too high..."

You: "What are the symptoms?"  ← AI knows you're still asking about diabetes
AI: "The symptoms of diabetes include increased thirst, frequent urination..."

You: "How is it treated?"  ← AI maintains context about diabetes
AI: "Diabetes treatment involves blood sugar monitoring, medication..."
```

## 📁 Project Structure

```
MedicalRag/
├── app/
│   ├── application.py              # 🌐 Main Flask web application
│   ├── requirements.txt            # 📦 Python dependencies
│   ├── .env.example               # 🔧 Environment variables template
│   │
│   ├── components/                 # 🔧 Core RAG Components
│   │   ├── llm.py                 # 🤖 OpenAI GPT integration
│   │   ├── embeddings.py          # 📊 OpenAI embeddings (text-embedding-3-small)
│   │   ├── memory.py              # 🧠 Advanced conversation memory system
│   │   ├── vector_store.py        # 🗄️ FAISS vector database management
│   │   ├── retriever.py           # 🔍 Document retrieval and QA chain
│   │   ├── pdf_loader.py          # 📄 PDF processing utilities
│   │   └── data_loader.py         # ⚡ FAISS index creation utility
│   │
│   ├── config/                     # ⚙️ Configuration
│   │   └── config.py              # 🔧 Application settings
│   │
│   ├── common/                     # 🛠️ Utilities
│   │   ├── logger.py              # 📝 Logging system
│   │   └── custom_exception.py    # ❗ Exception handling
│   │
│   ├── templates/                  # 🎨 Web Interface
│   │   └── index.html             # 💬 Modern chat UI
│   │
│   ├── data/                      # 📚 Medical Documents
│   │   └── *.pdf                  # Your medical PDF files
│   │
│   └── vectorstore/               # 🗃️ Vector Database
│       └── db_faiss/              # FAISS index files
│           ├── index.faiss
│           └── index.pkl
└── README.md                      # 📖 This file
```

## ⚙️ Configuration

### Model Settings (`config/config.py`)
```python
OPEN_AI_MODEL = "gpt-4o-mini"      # OpenAI model (fast, cost-effective)
CHUNK_SIZE = 500                   # Text chunk size for processing
CHUNK_OVERLAP = 50                 # Overlap between chunks for context
```

### Available OpenAI Models
- `gpt-4o-mini` ✅ (default - fast, economical)
- `gpt-4o` (more capable, higher cost)
- `gpt-4-turbo` (previous generation)

### Memory Configuration
- **Window Memory**: Keeps 3-6 recent message pairs
- **Summary Memory**: Summarizes conversations >10 messages
- **Auto-switching**: System automatically chooses optimal memory type

## 🔧 Usage Examples

### Basic Medical Queries
```
"What is hypertension?"
"Explain the symptoms of pneumonia"
"How is diabetes diagnosed?"
```

### Conversational Follow-ups
```
User: "What is asthma?"
AI: "Asthma is a respiratory condition..."

User: "What triggers it?"  ← AI knows "it" refers to asthma
AI: "Asthma can be triggered by allergens, exercise..."

User: "Any prevention methods?"  ← Continues context
AI: "To prevent asthma attacks, avoid known triggers..."
```

### UI Features
- **💾 Export Chat**: Download conversation as `.txt` file
- **🗑️ Clear History**: Reset conversation and memory
- **📊 Message Counter**: Track conversation length
- **🧠 Memory Status**: Visual indicator when smart memory is active

## 📚 Managing Medical Knowledge Base

### Adding New Documents

1. **Add PDF files** to the `data/` directory
2. **Recreate vector store**:
   ```bash
   conda activate medical-rag
   python components/data_loader.py
   ```
3. **Restart application** to use updated knowledge base

### Supported Document Types
- ✅ PDF files (medical textbooks, research papers, clinical guidelines)
- ✅ Text-based PDFs (searchable content)
- ❌ Image-only PDFs (not supported)

## 🛠️ Development & Troubleshooting

### Development Mode
```bash
conda activate medical-rag
export FLASK_DEBUG=1  # Enable debug mode
python application.py
```

### Common Issues & Solutions

**🔑 OpenAI API Error**
```
Error: API connection error
```
**Solution**: 
- Verify `OPENAI_API_KEY` in `.env` file
- Check OpenAI account credits
- Ensure stable internet connection

**🗄️ Vector Store Not Found**
```
Error: No vector store found
```
**Solution**:
```bash
python components/data_loader.py
```

**📄 PDF Processing Error**
```
Error: No documents loaded from PDFs
```
**Solution**:
- Ensure PDF files are in `data/` directory
- Verify PDFs are readable (not corrupted/password-protected)
- Check PDF contains extractable text

**🧠 Memory Issues**
```
Error: Failed to create conversational memory
```
**Solution**:
- Verify all dependencies installed: `pip install -r requirements.txt`
- Check OpenAI API key validity
- Restart application

### Dependencies (`requirements.txt`)
```txt
langchain==0.3.27
langchain_community==0.3.29
langchain-openai==0.3.32
faiss-cpu==1.12.0
pypdf==6.0.0
flask==3.1.2
python-dotenv==1.1.1
```

## 🚀 Deployment

### Production Environment Variables
```bash
OPENAI_API_KEY=your_production_api_key
FLASK_ENV=production
```

### Docker Deployment
```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY app/ .

# Expose port
EXPOSE 5000

# Run application
CMD ["python", "application.py"]
```

### Build and Run Docker Container
```bash
# Build image
docker build -t medical-rag-chatbot .

# Run container
docker run -p 5000:5000 -e OPENAI_API_KEY=your_key medical-rag-chatbot
```

## 🔬 Technical Architecture

### RAG Pipeline Flow
1. **📄 Document Processing**: PDFs → Text Chunks
2. **🔢 Embeddings**: Text → OpenAI Embeddings (1536 dimensions)
3. **🗄️ Vector Storage**: Embeddings → FAISS Index
4. **🔍 Query Processing**: Question → Similarity Search → Context Retrieval
5. **🤖 Response Generation**: Context + Memory + Question → GPT Response
6. **🧠 Memory Update**: Store conversation turn for future context

### Memory Architecture
- **Session Storage**: Flask sessions store message history
- **Dynamic Memory**: Creates fresh memory context per request
- **Context Population**: Rebuilds conversational context from session
- **Smart Switching**: Chooses memory type based on conversation length

## 📊 Performance & Costs

### OpenAI API Costs (Approximate)
- **GPT-4o-mini**: ~$0.0001 per 1K input tokens
- **Embeddings**: ~$0.00002 per 1K tokens
- **Average conversation**: $0.01-0.05 depending on length

### Performance Metrics
- **Response time**: 2-5 seconds (depending on context length)
- **Memory efficiency**: Automatic summarization for long conversations
- **Vector search**: Sub-second similarity search with FAISS

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes and test thoroughly
4. Commit: `git commit -m 'Add amazing feature'`
5. Push: `git push origin feature/amazing-feature`
6. Open a Pull Request

### Development Guidelines
- Follow existing code structure and patterns
- Add comprehensive error handling
- Update README.md for new features
- Test with various medical documents
- Ensure memory system compatibility

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **[OpenAI](https://openai.com/)** for GPT models and embeddings API
- **[LangChain](https://langchain.com/)** for RAG framework and memory systems
- **[FAISS](https://github.com/facebookresearch/faiss)** for efficient vector similarity search
- **[Flask](https://flask.palletsprojects.com/)** for web application framework

## 📞 Support

For support and questions:
- 🐛 **Bug Reports**: Create an issue on GitHub
- 💡 **Feature Requests**: Open a discussion on GitHub
- 📚 **Documentation**: Check this README and code comments
- 🔧 **Configuration Help**: Review the troubleshooting section

---

## ⚠️ Medical Disclaimer

**Important**: This chatbot is designed for **informational and educational purposes only**. It should **never be used as a substitute** for professional medical advice, diagnosis, or treatment. 

- Always consult qualified healthcare providers for medical decisions
- In case of medical emergencies, contact emergency services immediately
- The AI responses are based on training data and may contain errors
- Medical knowledge evolves rapidly; always verify information with current sources

---

**Version 1.0** | Built with ❤️ for medical education and research | Powered by OpenAI & Python 3.12