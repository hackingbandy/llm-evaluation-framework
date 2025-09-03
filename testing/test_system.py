#!/usr/bin/env python3
# =============================================================================
# Financial Advisor RAG System Test (EU AI Act Website)
# =============================================================================

import os
import sys
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
from financial_advisor_rag import FinancialAdvisorRAG, load_metrics_config

def test_imports():
    """Test if all required modules can be imported."""
    print("🔍 Testing imports...")
    
    try:
        import pandas as pd
        import numpy as np
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        import requests
        from bs4 import BeautifulSoup
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("📋 Run: pip install -r requirements.txt")
        return False

def test_metrics_config():
    """Test metrics configuration loading."""
    print("\n🔍 Testing metrics configuration...")
    
    try:
        config = load_metrics_config()
        
        # Check if config has required structure
        if "evaluation_dimensions" in config and "evaluation_settings" in config:
            print("✅ Metrics configuration loaded successfully")
            print(f"   Dimensions: {len(config['evaluation_dimensions'])}")
            print(f"   Model: {config['evaluation_settings']['evaluation_model']}")
            
            # Check if all metrics have prompts
            total_metrics = 0
            metrics_with_prompts = 0
            for dimension, dim_config in config['evaluation_dimensions'].items():
                for metric_name, metric_config in dim_config['metrics'].items():
                    total_metrics += 1
                    if 'prompt' in metric_config:
                        metrics_with_prompts += 1
            
            print(f"   Metrics: {metrics_with_prompts}/{total_metrics} have LLM prompts")
            return True
        else:
            print("❌ Invalid metrics configuration structure")
            return False
            
    except Exception as e:
        print(f"❌ Metrics configuration error: {e}")
        return False

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
        ("catalog.csv", "file"),
        (".env", "file")
    ]
    
    optional_files = [
        ("metrics_config.json", "file")
    ]
    
    all_good = True
    for file_path, file_type in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path} ({file_type}) exists")
        else:
            print(f"❌ {file_path} ({file_type}) missing")
            all_good = False
    
    for file_path, file_type in optional_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path} ({file_type}) exists")
        else:
            print(f"⚠️ {file_path} ({file_type}) missing (will use defaults)")
    
    return all_good

def test_eu_ai_act_website():
    """Test EU AI Act website accessibility."""
    print("\n🔍 Testing EU AI Act website...")
    
    try:
        import requests
        from bs4 import BeautifulSoup
        
        url = "https://eur-lex.europa.eu/eli/reg/2024/1689/oj/eng"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Check if we can find EU AI Act content
        main_content = soup.find('div', {'class': 'text'})
        if main_content:
            print("✅ EU AI Act website accessible")
            print(f"   Content length: {len(main_content.get_text())} characters")
            return True
        else:
            print("⚠️ EU AI Act website accessible but content structure unclear")
            return True  # Not a critical failure
            
    except Exception as e:
        print(f"❌ EU AI Act website test failed: {e}")
        print("📋 Will use fallback content")
        return True  # Not a critical failure

def test_rag_system():
    """Test RAG system with EU AI Act content."""
    print("\n🔍 Testing RAG system...")
    
    try:
        rag = FinancialAdvisorRAG()
        rag.load_documents()
        
        if rag.docs and len(rag.docs) > 0:
            print(f"✅ Loaded {len(rag.docs)} EU AI Act chunks")
            
            # Test with a simple question
            test_question = "What are the transparency requirements for AI systems?"
            relevant_docs = rag.get_relevant_docs(test_question)
            
            if relevant_docs:
                print(f"✅ Retrieved {len(relevant_docs)} relevant sections")
                return True
            else:
                print("❌ No relevant sections retrieved")
                return False
        else:
            print("❌ No EU AI Act content loaded")
            return False
            
    except Exception as e:
        print(f"❌ RAG system failed: {e}")
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

def test_llm_evaluation():
    """Test LLM-based evaluation functionality."""
    print("\n🔍 Testing LLM evaluation...")
    
    try:
        from financial_advisor_rag import LLMEvaluator
        
        evaluator = LLMEvaluator()
        test_question = "What are the EU AI Act requirements for transparency?"
        test_answer = "According to Article 5 of the EU AI Act, providers of AI systems must ensure that natural persons are informed that they are interacting with an AI system, unless this is obvious from the circumstances and the context of use."
        test_contexts = ["Article 5 - Transparency obligations: Providers of AI systems shall ensure that their systems are designed and developed in such a way that natural persons are informed that they are interacting with an AI system."]
        test_prompt = "Rate the transparency of the response according to EU AI Act Article 5."
        
        score = evaluator.evaluate_with_prompt(test_question, test_answer, test_contexts, test_prompt)
        
        if 0.0 <= score <= 1.0:
            print(f"✅ LLM evaluation working (score: {score:.3f})")
            return True
        else:
            print(f"❌ LLM evaluation returned invalid score: {score}")
            return False
            
    except Exception as e:
        print(f"❌ LLM evaluation failed: {e}")
        return False

def test_fallback_methods():
    """Test fallback evaluation methods."""
    print("\n🔍 Testing fallback methods...")
    
    try:
        from financial_advisor_rag import (
            keyword_check_evaluation, 
            pii_detection_evaluation, 
            bias_detection_evaluation,
            context_overlap_evaluation,
            length_analysis_evaluation,
            hallucination_detection_evaluation
        )
        
        # Test keyword check
        test_response = "According to Article 5 of the EU AI Act, providers must ensure transparency and risk management."
        score = keyword_check_evaluation(test_response, ["Article", "EU AI Act", "transparency", "risk"])
        print(f"   Keyword check: {score:.3f}")
        
        # Test PII detection
        score = pii_detection_evaluation(test_response)
        print(f"   PII detection: {score:.3f}")
        
        # Test bias detection
        score = bias_detection_evaluation(test_response)
        print(f"   Bias detection: {score:.3f}")
        
        # Test context overlap
        score = context_overlap_evaluation(test_response, ["Article 5 transparency obligations"])
        print(f"   Context overlap: {score:.3f}")
        
        # Test length analysis
        score = length_analysis_evaluation(test_response)
        print(f"   Length analysis: {score:.3f}")
        
        # Test hallucination detection
        score = hallucination_detection_evaluation(test_response, ["Article 5"])
        print(f"   Hallucination detection: {score:.3f}")
        
        print("✅ All fallback methods working")
        return True
        
    except Exception as e:
        print(f"❌ Fallback methods failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Financial Advisor RAG System Test (EU AI Act Website)")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Metrics Configuration", test_metrics_config),
        ("Environment", test_environment),
        ("Files", test_files),
        ("EU AI Act Website", test_eu_ai_act_website),
        ("RAG System", test_rag_system),
        ("Question Loading", test_question_loading),
        ("LLM Evaluation", test_llm_evaluation),
        ("Fallback Methods", test_fallback_methods)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
    
    print("\n" + "=" * 60)
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