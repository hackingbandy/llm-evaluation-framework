#!/usr/bin/env python3
# =============================================================================
# Financial Advisor RAG Evaluation Runner
# =============================================================================

import os
import sys
from financial_advisor_rag import main

def check_requirements():
    """Check if all required files exist."""
    required_files = [
        "financial_data/",
        "catalog.csv",
        ".env"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing required files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        print("\n📋 Setup instructions:")
        print("1. Create .env file: cp env_template.txt .env")
        print("2. Add your OpenAI API key to .env")
        print("3. Ensure financial_data/ contains PDF files")
        print("4. Ensure catalog.csv exists with questions")
        return False
    
    return True

def main_runner():
    """Main runner function with error handling."""
    print("🚀 Financial Advisor RAG Evaluation Runner")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Check OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not found in environment")
        print("   Please add your OpenAI API key to .env file")
        sys.exit(1)
    
    print("✅ All requirements met!")
    print("🔄 Starting evaluation...\n")
    
    try:
        # Run the main evaluation
        main()
        print("\n🎉 Evaluation completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⏹️ Evaluation interrupted by user")
        sys.exit(0)
        
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        print("📋 Troubleshooting:")
        print("1. Check your OpenAI API key")
        print("2. Ensure you have sufficient API credits")
        print("3. Check internet connection")
        sys.exit(1)

if __name__ == "__main__":
    main_runner() 