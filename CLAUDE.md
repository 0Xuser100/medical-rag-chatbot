# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Medical RAG (Retrieval-Augmented Generation) chatbot application that processes PDF documents and answers medical questions using LangChain and HuggingFace models.

## Architecture

The codebase follows a modular architecture with the following key components:

- **app/components/**: Core RAG pipeline components
  - `pdf_loader.py`: Loads and chunks PDF documents from data directory
  - `embeddings.py`: HuggingFace embeddings using sentence-transformers/all-MiniLM-L6-v2
  - `vector_store.py`: FAISS vector database management
  - `llm.py`: HuggingFace LLM integration (Mistral-7B-Instruct-v0.3)
  - `retriever.py`: QA chain with custom medical prompt template

- **app/config/**: Configuration management
  - `config.py`: Environment variables and constants

- **app/common/**: Utilities
  - `logger.py`: Logging setup
  - `custom_exception.py`: Exception handling

## Development Commands

### Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp app/.env.example app/.env
# Add your HuggingFace token to app/.env
```

### Data Processing
```bash
# Process PDFs and create vector store (run from project root)
python -m app.components.data_loader

# Alternative way to run data processing
python app/components/data_loader.py
```

### Configuration

Required environment variables in `app/.env`:
- `HF_TOKEN`: HuggingFace API token for model access
- `HUGGINGFACE_API_TOKEN`: Alternative token variable

Key configuration constants:
- Vector store path: `vectorstore/db_faiss`
- PDF data directory: `data/`
- Chunk size: 800 characters
- Chunk overlap: 100 characters

## Key Implementation Details

- Uses FAISS for vector storage with dangerous deserialization enabled
- Implements custom prompt template for medical Q&A with 2-3 line answers
- Retriever configured with k=4 for context documents
- LLM temperature set to 0.3 for consistent responses
- Text chunking uses RecursiveCharacterTextSplitter

## Data Flow

1. PDFs loaded from `data/` directory
2. Documents split into chunks (800 chars, 100 overlap)
3. Embeddings generated and stored in FAISS
4. User queries processed through retrieval chain
5. Context + question sent to Mistral LLM for response