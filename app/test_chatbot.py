#!/usr/bin/env python3
"""
Test script for Medical RAG Chatbot
This script tests all components independently to identify issues
"""

import os
import sys
from dotenv import load_dotenv

# Add app directory to path
sys.path.append('/mnt/d/LLMOPS/Medical-Rag/MedicalRag/app')

# Load environment variables
load_dotenv()

def test_config():
    """Test configuration loading"""
    print("=== Testing Configuration ===")
    try:
        from config.config import OPEN_AI_API_KEY, OPEN_AI_MODEL, DB_FAISS_PATH, DATA_PATH
        
        print(f"✓ OpenAI API Key loaded: {'Yes' if OPEN_AI_API_KEY else 'No'}")
        print(f"✓ OpenAI Model: {OPEN_AI_MODEL}")
        print(f"✓ FAISS Path: {DB_FAISS_PATH}")
        print(f"✓ Data Path: {DATA_PATH}")
        
        # Check if API key is set
        if not OPEN_AI_API_KEY:
            print("❌ ERROR: OPENAI_API_KEY not found in environment!")
            return False
            
        print("✅ Configuration test passed")
        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_vector_store():
    """Test vector store loading"""
    print("\n=== Testing Vector Store ===")
    try:
        from components.vector_store import load_vector_store
        
        db = load_vector_store()
        if db is None:
            print("❌ Vector store returned None")
            return False
            
        print("✓ Vector store loaded successfully")
        
        # Test similarity search
        results = db.similarity_search("What is diabetes?", k=3)
        print(f"✓ Similarity search returned {len(results)} documents")
        
        if results:
            print(f"✓ Sample result: {results[0].page_content[:100]}...")
            
        print("✅ Vector store test passed")
        return True
    except Exception as e:
        print(f"❌ Vector store test failed: {e}")
        return False

def test_embeddings():
    """Test embeddings model"""
    print("\n=== Testing Embeddings ===")
    try:
        from components.embeddings import get_embedding_model
        
        model = get_embedding_model()
        if model is None:
            print("❌ Embeddings model returned None")
            return False
            
        print("✓ Embeddings model loaded successfully")
        
        # Test embedding generation
        test_text = "This is a test"
        embeddings = model.embed_query(test_text)
        print(f"✓ Generated embeddings with dimension: {len(embeddings)}")
        
        print("✅ Embeddings test passed")
        return True
    except Exception as e:
        print(f"❌ Embeddings test failed: {e}")
        return False

def test_llm():
    """Test LLM loading"""
    print("\n=== Testing LLM ===")
    try:
        from components.llm import load_llm
        from config.config import OPEN_AI_MODEL, OPEN_AI_API_KEY
        
        llm = load_llm(OPEN_AI_MODEL, OPEN_AI_API_KEY)
        if llm is None:
            print("❌ LLM returned None")
            return False
            
        print("✓ LLM loaded successfully")
        
        # Test LLM invocation
        response = llm.invoke("Hello, this is a test. Please respond with 'Test successful'")
        print(f"✓ LLM response: {response.content[:100]}...")
        
        print("✅ LLM test passed")
        return True
    except Exception as e:
        print(f"❌ LLM test failed: {e}")
        return False

def test_retriever():
    """Test QA chain creation and execution"""
    print("\n=== Testing Retriever/QA Chain ===")
    try:
        from components.retriever import create_qa_chain
        
        qa_chain = create_qa_chain()
        if qa_chain is None:
            print("❌ QA chain returned None")
            return False
            
        print("✓ QA chain created successfully")
        
        # Test QA chain
        test_query = "What is diabetes?"
        response = qa_chain.invoke({"query": test_query})
        result = response.get("result", "No response")
        
        print(f"✓ Test query: {test_query}")
        print(f"✓ QA response: {result[:200]}...")
        
        print("✅ Retriever test passed")
        return True
    except Exception as e:
        print(f"❌ Retriever test failed: {e}")
        return False

def test_flask_app():
    """Test Flask application components"""
    print("\n=== Testing Flask Application ===")
    try:
        from application import app
        
        with app.test_client() as client:
            # Test GET request
            response = client.get('/')
            print(f"✓ GET / status: {response.status_code}")
            
            if response.status_code == 200:
                print("✓ Home page loads successfully")
            
            # Test POST request
            response = client.post('/', data={'prompt': 'What is diabetes?'})
            print(f"✓ POST / status: {response.status_code}")
            
        print("✅ Flask application test passed")
        return True
    except Exception as e:
        print(f"❌ Flask application test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🏥 Medical RAG Chatbot Test Suite")
    print("=" * 50)
    
    tests = [
        test_config,
        test_embeddings,
        test_vector_store,
        test_llm,
        test_retriever,
        test_flask_app
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    for i, (test, result) in enumerate(zip(tests, results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{i+1}. {test.__name__}: {status}")
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Chatbot should work correctly.")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    main()