# Financial Advisor RAG System with EU AI Act Website Integration

A comprehensive RAG (Retrieval-Augmented Generation) system for financial advice that uses the official EU AI Act website as its knowledge source and evaluates responses using LLM-based evaluation for EU AI Act compliance.

## Features

- **EU AI Act Integration**: Directly loads content from the official EU AI Act website
- **Real-time Content**: Always uses the latest version of Regulation (EU) 2024/1689
- **Question Catalog**: Uses questions from `catalog.csv` for evaluation
- **LLM-Based Evaluation**: Custom evaluation for EU AI Act compliance metrics
- **Configurable Metrics**: JSON-based configuration for easy customization
- **Fallback Content**: Comprehensive fallback if website is unavailable

## Quick Start

1. **Setup Environment**:
   ```bash
   pip install -r requirements.txt
   cp env_template.txt .env
   # Add your OpenAI API key to .env
   ```

2. **Prepare Data**:
   - Ensure `catalog.csv` contains questions and categories
   - System automatically loads EU AI Act content from website

3. **Run Tests**:
   ```bash
   python test_system.py
   ```

4. **Run Evaluation**:
   ```bash
   python financial_advisor_rag.py
   ```

## EU AI Act Integration

The system directly integrates with the official EU AI Act website:
- **Source**: https://eur-lex.europa.eu/eli/reg/2024/1689/oj/eng
- **Content**: Full Regulation (EU) 2024/1689 text
- **Articles**: All 27 articles of the EU AI Act
- **Annexes**: Complete regulatory annexes
- **Fallback**: Comprehensive offline content if website unavailable

## Configuration

The system uses `metrics_config.json` for configuration:

```json
{
  "evaluation_dimensions": {
    "accuracy_performance": {
      "description": "Assesses correctness, relevance, and usefulness of responses",
      "metrics": {
        "context_relevancy": {
          "description": "Assesses if retrieved EU AI Act sections are relevant",
          "prompt": "Rate how relevant the retrieved EU AI Act sections are...",
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

### Accuracy Performance
- **Context Relevancy**: Assesses if retrieved EU AI Act sections are relevant
- **Answer Relevancy**: Assesses if the answer is relevant to EU AI Act questions
- **Faithfulness**: Determines if response is grounded in EU AI Act context

### EU AI Act Compliance
- **Transparency**: AI identification per Article 5
- **Safety**: Risk assessment per Articles 6 and 7
- **Privacy**: PII protection per Article 9
- **Fairness**: Unbiased treatment per EU AI Act requirements
- **Accountability**: Responsibility per Articles 7, 8, and 10

### Fallback Methods
When LLM evaluation is unavailable, the system uses:
- **Keyword Matching**: EU AI Act specific terms
- **PII Detection**: Pattern-based personal data detection
- **Bias Detection**: Bias indicator analysis
- **Context Overlap**: Word overlap with EU AI Act content
- **Hallucination Detection**: Detects non-EU AI Act claims

## File Structure

```
├── financial_advisor_rag.py    # Main system with EU AI Act integration
├── test_system.py              # Test suite for EU AI Act website
├── metrics_config.json         # Metrics configuration
├── requirements.txt            # Dependencies including web scraping
├── catalog.csv                 # Question catalog
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
- Internet connection (for EU AI Act website)
- Question catalog (catalog.csv)

## Troubleshooting

1. **Import Errors**: Run `pip install -r requirements.txt`
2. **API Key Issues**: Check `.env` file configuration
3. **Website Issues**: System automatically uses fallback content
4. **File Not Found**: Ensure all required files are in place

## Customization

- **Add Metrics**: Edit `metrics_config.json` to add new evaluation dimensions
- **Modify Prompts**: Update evaluation prompts in the configuration
- **Change Models**: Modify `evaluation_model` in settings
- **Adjust Thresholds**: Update score thresholds for different performance levels

## EU AI Act Compliance

The system evaluates compliance with Regulation (EU) 2024/1689 requirements:
- **Article 5**: Transparency obligations
- **Article 6**: Accuracy, robustness and cybersecurity
- **Article 7**: Human oversight
- **Article 8**: Fundamental rights impact assessment
- **Article 9**: Data governance and management
- **Article 10**: Documentation and record keeping
- **Article 16**: Right to explanation
- **Article 18**: Right to compensation

## License

This project is for educational and research purposes. Ensure compliance with applicable regulations when using in production environments.




