#!/usr/bin/env python3
"""
Recreate vector store with OpenAI embeddings
Run this if the existing FAISS store is incompatible
"""

import os
import sys
sys.path.append('/mnt/d/LLMOPS/Medical-Rag/MedicalRag/app')

from components.pdf_loader import load_pdf_files, create_text_chunks
from components.vector_store import save_vector_store
from config.config import DATA_PATH, DB_FAISS_PATH
import shutil

def main():
    print("🔄 Recreating vector store with OpenAI embeddings...")
    
    try:
        # Check if data directory exists
        if not os.path.exists(DATA_PATH):
            print(f"❌ Data directory not found: {DATA_PATH}")
            return False
        
        # Find PDF files
        pdf_files = [f for f in os.listdir(DATA_PATH) if f.endswith('.pdf')]
        
        if not pdf_files:
            print(f"❌ No PDF files found in {DATA_PATH}")
            return False
        
        print(f"📄 Found PDF files: {', '.join(pdf_files)}")
        
        # Load PDF documents
        print("📝 Loading PDF documents...")
        documents = load_pdf_files()
        
        if not documents:
            print("❌ No documents loaded from PDFs")
            return False
        
        print(f"✓ Loaded {len(documents)} document pages")
        
        # Create text chunks
        print("✂️  Creating text chunks...")
        text_chunks = create_text_chunks(documents)
        
        if not text_chunks:
            print("❌ No text chunks extracted")
            return False
            
        print(f"✓ Extracted {len(text_chunks)} text chunks")
        
        # Remove old vector store if exists
        if os.path.exists(DB_FAISS_PATH):
            print("🗑️  Removing old vector store...")
            shutil.rmtree(DB_FAISS_PATH)
        
        # Create new vector store
        print("🏗️  Creating new vector store with OpenAI embeddings...")
        db = save_vector_store(text_chunks)
        
        if db:
            print("✅ Vector store created successfully!")
            
            # Test the new vector store
            print("🧪 Testing vector store...")
            results = db.similarity_search("What is diabetes?", k=3)
            print(f"✓ Test search returned {len(results)} results")
            
            return True
        else:
            print("❌ Failed to create vector store")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Vector store recreation completed successfully!")
        print("You can now run the chatbot application.")
    else:
        print("\n💥 Vector store recreation failed!")
        print("Please check the errors above.")