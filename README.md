# Financial Advisor RAG System with Comprehensive Evaluation

A comprehensive RAG (Retrieval-Augmented Generation) system for financial advice that evaluates responses using both RAGAS metrics and LLM-based evaluation for EU AI Act compliance.

## Features

- **Document Processing**: Loads and processes financial PDF documents
- **Question Catalog**: Uses questions from `catalog.csv` for evaluation
- **RAGAS Evaluation**: Standard RAG metrics (context_relevancy, answer_relevancy, faithfulness)
- **LLM-Based Evaluation**: Custom evaluation for EU AI Act compliance metrics
- **Configurable Metrics**: JSON-based configuration for easy customization
- **Fallback Evaluation**: Basic evaluation when RAGAS is unavailable

## Quick Start

1. **Setup Environment**:
   ```bash
   pip install -r requirements.txt
   cp env_template.txt .env
   # Add your OpenAI API key to .env
   ```

2. **Prepare Data**:
   - Add financial PDF documents to `financial_data/`
   - Ensure `catalog.csv` contains questions and categories

3. **Run Tests**:
   ```bash
   python test_system.py
   ```

4. **Run Evaluation**:
   ```bash
   python financial_advisor_rag.py
   ```

## Configuration

The system uses `metrics_config.json` for configuration:

```json
{
  "evaluation_dimensions": {
    "accuracy_performance": {
      "description": "Assesses correctness, relevance, and usefulness of responses",
      "metrics": {
        "context_relevancy": {
          "ragas_metric": "context_relevancy",
          "description": "Assesses if retrieved contexts are relevant to the question",
          "prompt": "Rate how relevant the retrieved documents are...",
          "fallback": "keyword_matching"
        }
      }
    }
  },
  "evaluation_settings": {
    "max_questions": 8,
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "evaluation_model": "gpt-4o-mini"
  }
}
```

## Evaluation Metrics

### RAGAS Metrics (when available)
- **Context Relevancy**: Assesses if retrieved contexts are relevant
- **Answer Relevancy**: Assesses if the answer is relevant to the question
- **Faithfulness**: Determines if response is grounded in provided context

### LLM-Based Metrics (EU AI Act Compliance)
- **Transparency**: AI identification and capability disclosure
- **Safety**: Risk assessment and safety measures
- **Privacy**: PII protection and data privacy
- **Fairness**: Unbiased treatment across demographic groups
- **Accountability**: Clear responsibility and oversight mechanisms

### Fallback Methods
When RAGAS metrics are unavailable, the system uses:
- **Keyword Matching**: Simple keyword-based evaluation
- **PII Detection**: Pattern-based personal data detection
- **Bias Detection**: Bias indicator analysis
- **Context Overlap**: Word overlap analysis

## File Structure

```
├── financial_advisor_rag.py    # Main system
├── test_system.py              # Test suite
├── metrics_config.json         # Metrics configuration
├── requirements.txt            # Dependencies
├── catalog.csv                 # Question catalog
├── financial_data/             # PDF documents
├── .env                        # Environment variables
└── README.md                   # This file
```

## Output

The system generates:
- **Console Output**: Real-time evaluation progress and results
- **evaluation_results.txt**: Detailed evaluation report
- **Status Indicators**: ✅ Excellent, ✅ Good, ⚠️ Acceptable, ❌ Needs Improvement

## Requirements

- Python 3.8+
- OpenAI API key
- Financial PDF documents
- Question catalog (catalog.csv)

## Troubleshooting

1. **Import Errors**: Run `pip install -r requirements.txt`
2. **API Key Issues**: Check `.env` file configuration
3. **RAGAS Issues**: System falls back to LLM-based evaluation
4. **File Not Found**: Ensure all required files are in place

## Customization

- **Add Metrics**: Edit `metrics_config.json` to add new evaluation dimensions
- **Modify Prompts**: Update evaluation prompts in the configuration
- **Change Models**: Modify `evaluation_model` in settings
- **Adjust Thresholds**: Update score thresholds for different performance levels

## EU AI Act Compliance

The system evaluates compliance with EU AI Act (Regulation EU 2024/1689) requirements:
- **Transparency**: Clear AI identification and capability disclosure
- **Safety**: Risk assessment and appropriate safety measures
- **Privacy**: Protection of personal data and PII
- **Fairness**: Unbiased treatment across all demographic groups
- **Accountability**: Clear responsibility assignment and oversight

## License

This project is for educational and research purposes. Ensure compliance with applicable regulations when using in production environments.




