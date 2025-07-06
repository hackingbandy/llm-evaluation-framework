# LLM Evaluation Framework

**Python 3.12 Compatible**

Port: http://127.0.0.1:8000
Start command: `uvicorn main:app --reload`

## Setup for Python 3.12

### 1. Create Python 3.12 Environment
```bash
# Using venv
python3.12 -m venv venv
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows

# Using conda
conda create -n llm-eval python=3.12
conda activate llm-eval
```

### 2. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Environment Variables
Create a `.env` file in the project root:
```
OPENAI_API_KEY=your_openai_api_key_here
USER_AGENT=LLM-Evaluation-Framework/1.0
```

### 4. Run the Application
```bash
uvicorn main:app --reload
```

## Features
- **EU AI Act Compliant** LLM responses
- **Financial Advisor** with Yahoo Finance integration
- **RAG System** with validation CSV support
- **BLEU Score** and semantic similarity analysis
- **Real-time Chat** interface
- **File Upload** for CSV validation

## Python 3.12 Benefits
- Better performance
- Improved error messages
- Enhanced type hints
- Latest security updates
- Better SSL/TLS support (resolves urllib3 warnings)

# To-Do's
- [x] clean the project
- [ ] read https://python.langchain.com/docs/tutorials/rag/
- [ ] build a RAG-System
- [ ] implement ragas to evaluate the response
- [ ] read https://docs.ragas.io/en/stable/getstarted/evals/



