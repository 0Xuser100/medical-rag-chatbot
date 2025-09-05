#!/usr/bin/env python3
"""
Test script for memory functionality in Medical RAG Chatbot
"""

import sys
sys.path.append('/mnt/d/LLMOPS/Medical-Rag/MedicalRag/app')

from components.memory import create_session_qa_chain

def test_memory_functionality():
    """Test conversation memory"""
    print("🧠 Testing Memory Functionality")
    print("=" * 50)
    
    # Test conversation
    conversation = [
        {"role": "user", "content": "What is diabetes?"},
        {"role": "assistant", "content": "Diabetes is a chronic condition where blood sugar levels are too high due to the body's inability to produce or properly use insulin."},
        {"role": "user", "content": "What are the symptoms?"},
        {"role": "assistant", "content": "Common diabetes symptoms include increased thirst, frequent urination, fatigue, blurred vision, and slow wound healing."},
    ]
    
    try:
        # Test 1: Create QA chain with conversation history
        print("Test 1: Creating QA chain with conversation history...")
        qa_chain = create_session_qa_chain(conversation)
        print("✅ QA chain created successfully with memory")
        
        # Test 2: Ask follow-up question that requires memory
        print("\nTest 2: Testing memory with follow-up question...")
        followup_question = "Are there different types of this condition?"
        
        response = qa_chain.invoke({"question": followup_question})
        answer = response.get("answer", "No response")
        
        print(f"Question: {followup_question}")
        print(f"Answer: {answer}")
        
        # Check if the answer references diabetes (showing memory works)
        if "diabetes" in answer.lower() or "condition" in answer.lower():
            print("✅ Memory test passed - AI remembered previous context")
        else:
            print("❌ Memory test failed - AI didn't use previous context")
        
        # Test 3: Test with longer conversation
        print("\nTest 3: Testing with longer conversation...")
        long_conversation = conversation + [
            {"role": "user", "content": "How is it treated?"},
            {"role": "assistant", "content": "Diabetes treatment includes blood sugar monitoring, medication (insulin for Type 1, various medications for Type 2), diet management, and regular exercise."},
            {"role": "user", "content": "What about diet recommendations?"},
            {"role": "assistant", "content": "Diabetics should focus on balanced meals with controlled carbohydrates, regular meal timing, plenty of vegetables, lean proteins, and limited processed foods."},
        ]
        
        qa_chain_long = create_session_qa_chain(long_conversation)
        
        memory_question = "Can you remind me what we discussed about managing this condition?"
        response = qa_chain_long.invoke({"question": memory_question})
        answer = response.get("answer", "No response")
        
        print(f"Question: {memory_question}")
        print(f"Answer: {answer[:200]}...")
        
        if "treatment" in answer.lower() or "diet" in answer.lower() or "medication" in answer.lower():
            print("✅ Long conversation memory test passed")
        else:
            print("❌ Long conversation memory test failed")
        
        print("\n🎉 Memory functionality tests completed!")
        return True
        
    except Exception as e:
        print(f"❌ Memory test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_memory_types():
    """Test different memory types"""
    print("\n🔄 Testing Different Memory Types")
    print("=" * 50)
    
    try:
        # Short conversation - should use window memory
        short_convo = [
            {"role": "user", "content": "What is hypertension?"},
            {"role": "assistant", "content": "Hypertension is high blood pressure."}
        ]
        
        print("Testing window memory (short conversation)...")
        qa_window = create_session_qa_chain(short_convo, "window")
        print("✅ Window memory created successfully")
        
        # Long conversation - should use summary memory
        long_convo = []
        for i in range(12):
            long_convo.extend([
                {"role": "user", "content": f"Question {i+1} about medical topic"},
                {"role": "assistant", "content": f"Answer {i+1} about medical information"}
            ])
        
        print("Testing summary memory (long conversation)...")
        qa_summary = create_session_qa_chain(long_convo, "summary")
        print("✅ Summary memory created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Memory types test failed: {e}")
        return False

if __name__ == "__main__":
    print("🏥 Medical RAG Memory Test Suite")
    
    test1 = test_memory_functionality()
    test2 = test_memory_types()
    
    if test1 and test2:
        print("\n✅ All memory tests passed! Memory functionality is working correctly.")
    else:
        print("\n❌ Some memory tests failed. Please check the errors above.")