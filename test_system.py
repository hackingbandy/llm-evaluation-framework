#!/usr/bin/env python3
# =============================================================================
# Financial Advisor RAG System Test
# =============================================================================

import os
import sys
from dotenv import load_dotenv
from financial_advisor_rag import FinancialAdvisorRAG, evaluate_eu_ai_act_compliance

def test_imports():
    """Test if all required modules can be imported."""
    print("🔍 Testing imports...")
    
    try:
        import pandas as pd
        import numpy as np
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from langchain_community.document_loaders import PyPDFLoader
        from ragas import evaluate, EvaluationDataset
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("📋 Run: pip install -r requirements.txt")
        return False

def test_ragas_metrics():
    """Test if RAGAS metrics are available."""
    print("\n🔍 Testing RAGAS metrics...")
    
    try:
        from ragas.metrics import faithfulness, answer_relevancy, context_relevancy
        print("✅ RAGAS metrics available")
        return True
    except ImportError:
        print("⚠️ RAGAS metrics not available, will use basic evaluation")
        return True  # This is not a critical failure

def test_environment():
    """Test environment variables."""
    print("\n🔍 Testing environment...")
    
    load_dotenv()
    
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not found")
        print("📋 Add your API key to .env file")
        return False
    
    print("✅ Environment variables OK")
    return True

def test_files():
    """Test if required files exist."""
    print("\n🔍 Testing files...")
    
    required_files = [
        ("financial_data/", "directory"),
        ("catalog.csv", "file"),
        (".env", "file")
    ]
    
    all_good = True
    for file_path, file_type in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path} ({file_type}) exists")
        else:
            print(f"❌ {file_path} ({file_type}) missing")
            all_good = False
    
    return all_good

def test_pdf_loading():
    """Test PDF loading functionality."""
    print("\n🔍 Testing PDF loading...")
    
    try:
        rag = FinancialAdvisorRAG()
        rag.load_documents()
        
        if rag.docs and len(rag.docs) > 0:
            print(f"✅ Loaded {len(rag.docs)} document chunks")
            return True
        else:
            print("❌ No documents loaded")
            return False
            
    except Exception as e:
        print(f"❌ PDF loading failed: {e}")
        return False

def test_question_loading():
    """Test question loading from catalog."""
    print("\n🔍 Testing question loading...")
    
    try:
        import pandas as pd
        df = pd.read_csv("catalog.csv")
        
        if 'Question' in df.columns and len(df) > 0:
            print(f"✅ Loaded {len(df)} questions from catalog.csv")
            return True
        else:
            print("❌ No questions found in catalog.csv")
            return False
            
    except Exception as e:
        print(f"❌ Question loading failed: {e}")
        return False

def test_eu_ai_act_evaluation():
    """Test EU AI Act compliance evaluation."""
    print("\n🔍 Testing EU AI Act evaluation...")
    
    test_response = """
    As an AI financial advisor, I can provide general guidance on portfolio diversification.
    However, I recommend consulting with a qualified financial professional for personalized advice.
    Please note that all investments carry risk and past performance does not guarantee future results.
    """
    
    try:
        scores = evaluate_eu_ai_act_compliance(test_response)
        
        if all(criterion in scores for criterion in ["transparency", "fairness", "safety", "privacy", "accountability"]):
            print("✅ EU AI Act evaluation working")
            print(f"   Sample scores: {scores}")
            return True
        else:
            print("❌ EU AI Act evaluation incomplete")
            return False
            
    except Exception as e:
        print(f"❌ EU AI Act evaluation failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Financial Advisor RAG System Test")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("RAGAS Metrics", test_ragas_metrics),
        ("Environment", test_environment),
        ("Files", test_files),
        ("PDF Loading", test_pdf_loading),
        ("Question Loading", test_question_loading),
        ("EU AI Act Evaluation", test_eu_ai_act_evaluation)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready to run.")
        print("🚀 Run: python financial_advisor_rag.py")
    else:
        print("⚠️ Some tests failed. Please fix issues before running.")
        print("📋 Check the error messages above for guidance.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 