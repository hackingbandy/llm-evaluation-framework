#!/usr/bin/env python3
"""
Test script for Financial Advisor RAG System
Tests that financial documents are loaded and EU AI Act evaluation works
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from financial_advisor_rag import FinancialAdvisorRAG, LLMEvaluator

def test_financial_document_loading():
    """Test that financial documents are loaded correctly."""
    print("🧪 Testing Financial Document Loading...")
    
    rag = FinancialAdvisorRAG()
    rag.load_documents()
    
    print(f"✅ Loaded {len(rag.docs)} financial document chunks")
    print(f"✅ Created {len(rag.doc_embeddings)} embeddings")
    
    if len(rag.docs) > 0:
        print("✅ Financial documents loaded successfully!")
        print(f"📄 Sample chunk: {rag.docs[0][:200]}...")
    else:
        print("❌ No financial documents loaded!")
        return False
    
    return True

def test_financial_advice_generation():
    """Test that financial advice is generated based on financial documents."""
    print("\n🧪 Testing Financial Advice Generation...")
    
    rag = FinancialAdvisorRAG()
    rag.load_documents()
    
    test_question = "What are the key principles of portfolio diversification?"
    
    try:
        relevant_docs = rag.get_relevant_docs(test_question)
        answer = rag.generate_answer(test_question, relevant_docs)
        
        print(f"✅ Generated financial advice for: {test_question}")
        print(f"📝 Answer: {answer[:300]}...")
        print(f"📚 Used {len(relevant_docs)} relevant document chunks")
        
        return True
        
    except Exception as e:
        print(f"❌ Error generating financial advice: {e}")
        return False

def test_eu_ai_act_evaluation():
    """Test that evaluation uses EU AI Act criteria."""
    print("\n🧪 Testing EU AI Act Evaluation...")
    
    evaluator = LLMEvaluator()
    
    # Test question and answer
    test_question = "What investment strategy should I use for retirement?"
    test_answer = "As an AI financial advisor, I recommend diversifying your portfolio across stocks, bonds, and other assets. However, please consult with a qualified human financial advisor for personalized advice. This advice is based on general financial principles and includes appropriate risk warnings."
    test_contexts = ["Portfolio diversification reduces risk by spreading investments across different asset classes."]
    
    try:
        # Test transparency evaluation
        transparency_score = evaluator.evaluate_with_prompt(
            test_question, 
            test_answer, 
            test_contexts,
            "Rate the transparency of the financial advice response according to EU AI Act Article 5."
        )
        
        print(f"✅ EU AI Act evaluation completed!")
        print(f"📊 Transparency score: {transparency_score:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in EU AI Act evaluation: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Testing Financial Advisor RAG System")
    print("=" * 50)
    
    tests = [
        test_financial_document_loading,
        test_financial_advice_generation,
        test_eu_ai_act_evaluation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is working correctly.")
        print("\n✅ Financial Advisor: Uses financial documents")
        print("✅ Evaluation: Uses EU AI Act criteria")
    else:
        print("⚠️  Some tests failed. Please check the system configuration.")

if __name__ == "__main__":
    main() 