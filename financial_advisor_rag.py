# =============================================================================
# LLM Evaluation Framework - Financial Advisor RAG System
# =============================================================================

# =============================================================================
# Environment Setup & Imports
# =============================================================================
import os
from typing import List
from dotenv import load_dotenv

# Set USER_AGENT immediately to avoid warnings
os.environ['USER_AGENT'] = 'LLM-Evaluation-Framework/1.0 (https://github.com/hackingbandy/llm-evaluation-framework)'

# LangChain imports
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import TypedDict

# Financial data imports
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# Web scraping
import bs4
from langchain import hub

# =============================================================================
# Environment Configuration
# =============================================================================
# Load environment variables
load_dotenv()

# Get API keys and configuration
openai_api_key = os.getenv("OPENAI_API_KEY")

# Verify USER_AGENT is set
user_agent = os.getenv("USER_AGENT")
if not user_agent:
    os.environ['USER_AGENT'] = 'LLM-Evaluation-Framework/1.0 (https://github.com/hackingbandy/llm-evaluation-framework)'
    print(f"USER_AGENT set to: {os.environ['USER_AGENT']}")
else:
    print(f"USER_AGENT already set: {user_agent}")

# =============================================================================
# Model Initialization
# =============================================================================
# Initialize LLM with financial advisor system prompt
llm = init_chat_model("gpt-4o-mini", model_provider="openai")

# Initialize Embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Initialize Vector Store
vector_store = InMemoryVectorStore(embeddings)

# =============================================================================
# Financial Data Sources & Processing
# =============================================================================
def get_financial_data(symbol: str = "AAPL", period: str = "1mo") -> str:
    """
    Get real-time financial data for analysis.
    """
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period)
        
        if hist.empty:
            return f"No data available for {symbol}"
        
        current_price = hist['Close'].iloc[-1]
        price_change = hist['Close'].iloc[-1] - hist['Close'].iloc[0]
        price_change_pct = (price_change / hist['Close'].iloc[0]) * 100
        
        info = ticker.info
        company_name = info.get('longName', symbol)
        sector = info.get('sector', 'Unknown')
        
        financial_summary = f"""
        Company: {company_name} ({symbol})
        Sector: {sector}
        Current Price: ${current_price:.2f}
        Price Change ({period}): ${price_change:.2f} ({price_change_pct:.2f}%)
        Market Cap: ${info.get('marketCap', 'N/A'):,}
        P/E Ratio: {info.get('trailingPE', 'N/A')}
        Dividend Yield: {info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0:.2f}%
        """
        
        return financial_summary
        
    except Exception as e:
        return f"Error fetching data for {symbol}: {str(e)}"

def get_portfolio_examples() -> str:
    """
    Generate example portfolios with real financial data.
    """
    portfolio_examples = []
    
    # Conservative Portfolio
    conservative_stocks = ["VTI", "BND", "VXUS"]
    conservative_data = []
    for stock in conservative_stocks:
        conservative_data.append(get_financial_data(stock, "1mo"))
    
    # Growth Portfolio
    growth_stocks = ["QQQ", "VGT", "ARKK"]
    growth_data = []
    for stock in growth_stocks:
        growth_data.append(get_financial_data(stock, "1mo"))
    
    portfolio_examples.append("""
    CONSERVATIVE PORTFOLIO (Low Risk):
    - 40% VTI (Total Stock Market ETF)
    - 40% BND (Total Bond Market ETF) 
    - 20% VXUS (International Stocks ETF)
    
    GROWTH PORTFOLIO (Higher Risk):
    - 50% QQQ (NASDAQ-100 ETF)
    - 30% VGT (Technology ETF)
    - 20% ARKK (Innovation ETF)
    """)
    
    return "\n".join(portfolio_examples)

# =============================================================================
# Data Loading & Processing
# =============================================================================
# Load financial education content
financial_sources = [
    "https://www.investopedia.com/terms/i/investment.asp",
    "https://www.investopedia.com/terms/d/diversification.asp",
    "https://www.investopedia.com/terms/r/riskmanagement.asp",
    "https://www.investopedia.com/terms/e/etf.asp"
]

print("Loading financial education content...")
all_docs = []

for source in financial_sources:
    try:
        loader = WebBaseLoader(web_paths=(source,))
        docs = loader.load()
        all_docs.extend(docs)
        print(f"✅ Loaded: {source}")
    except Exception as e:
        print(f"❌ Failed to load {source}: {e}")

# Add financial data as documents
financial_data_doc = Document(
    page_content=get_portfolio_examples(),
    metadata={"source": "financial_data", "type": "portfolio_examples"}
)
all_docs.append(financial_data_doc)

# Verify documents loaded correctly
print(f"Total documents loaded: {len(all_docs)}")

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # chunk size (characters)
    chunk_overlap=200,    # chunk overlap (characters)
    add_start_index=True, # track index in original document
)
all_splits = text_splitter.split_documents(all_docs)

print(f"Split into {len(all_splits)} sub-documents.")

# Index chunks in vector store
document_ids = vector_store.add_documents(documents=all_splits)
print(f"Indexed {len(document_ids)} documents in vector store.")

# =============================================================================
# Financial Advisor Prompt Configuration
# =============================================================================
# EU AI Act compliant financial advisor prompt
financial_advisor_template = """Du bist ein EU AI Act konformer Finanzberater. 

WICHTIGE COMPLIANCE-REGELN:
1. Kennzeichne dich immer als KI-System
2. Gib eine spezifischen Anlageempfehlungen
3. Betone die Wichtigkeit der Diversifikation
4. Erwähne immer die Risiken von Investitionen
5. Empfehle professionelle Beratung für individuelle Situationen
6. Verwende nur öffentlich verfügbare Informationen

KONTEXT (Finanzwissen und aktuelle Daten):
{context}

FRAGE DES KUNDEN: {question}

ANTWORTE ALS FINANZBERATER:
- Begrüße den Kunden professionell
- Verwende den bereitgestellten Kontext für fundierte Antworten
- Integriere aktuelle Marktdaten wenn relevant
- Betone Risikomanagement und Diversifikation
- Schließe mit einem Hinweis auf professionelle Beratung
- Verwende maximal 4-5 Sätze für eine präzise Antwort

Beispiel-Format:
"Guten Tag! Als KI-Finanzberater kann ich Ihnen Informationen zu [Thema] geben. Basierend auf den aktuellen Marktdaten... [Antwort mit Kontext]. Wichtig ist dabei immer die Diversifikation und das Bewusstsein für Risiken.
"

Antwort:"""

financial_advisor_prompt = PromptTemplate.from_template(financial_advisor_template)

# =============================================================================
# Type Definitions
# =============================================================================
class FinancialState(TypedDict):
    """Application state for the Financial Advisor RAG graph."""
    question: str
    context: List[Document]
    answer: str
    financial_data: str

# =============================================================================
# Graph Functions
# =============================================================================
def retrieve_financial_context(state: FinancialState) -> dict:
    """
    Retrieve relevant financial documents based on the question.
    """
    retrieved_docs = vector_store.similarity_search(state["question"], k=3)
    return {"context": retrieved_docs}

def get_relevant_financial_data(state: FinancialState) -> dict:
    """
    Get relevant real-time financial data based on the question.
    """
    question_lower = state["question"].lower()
    financial_data = ""
    
    # Extract potential stock symbols or add relevant data
    if any(word in question_lower for word in ["apple", "aapl", "tech"]):
        financial_data = get_financial_data("AAPL", "1mo")
    elif any(word in question_lower for word in ["tesla", "tsla", "electric"]):
        financial_data = get_financial_data("TSLA", "1mo")
    elif any(word in question_lower for word in ["microsoft", "msft"]):
        financial_data = get_financial_data("MSFT", "1mo")
    elif any(word in question_lower for word in ["portfolio", "diversification", "invest"]):
        financial_data = get_portfolio_examples()
    else:
        # Default to market overview
        financial_data = "Market Overview: Consider diversifying across different asset classes and sectors."
    
    return {"financial_data": financial_data}

def generate_financial_advice(state: FinancialState) -> dict:
    """
    Generate EU AI Act compliant financial advice based on retrieved context.
    """
    # Combine context and financial data
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    combined_context = f"{docs_content}\n\nAktuelle Finanzdaten:\n{state['financial_data']}"
    
    # Generate response using the financial advisor prompt
    messages = financial_advisor_prompt.invoke({
        "question": state["question"], 
        "context": combined_context
    })
    response = llm.invoke(messages)
    
    return {"answer": response.content}

# =============================================================================
# Graph Construction
# =============================================================================
# Build the Financial Advisor RAG graph
graph_builder = StateGraph(FinancialState)
graph_builder.add_node("retrieve", retrieve_financial_context)
graph_builder.add_node("get_data", get_relevant_financial_data)
graph_builder.add_node("generate", generate_financial_advice)

# Define the workflow
graph_builder.add_edge(START, "retrieve")
graph_builder.add_edge("retrieve", "get_data")
graph_builder.add_edge("get_data", "generate")

graph = graph_builder.compile()

# =============================================================================
# Test Execution
# =============================================================================
print("="*60)
print("FINANCIAL ADVISOR RAG SYSTEM TESTING")
print("="*60)

# Load questions from CSV file
def load_questions_from_csv(csv_file: str = "catalog.csv") -> List[str]:
    """
    Load test questions from CSV file.
    """
    try:
        df = pd.read_csv(csv_file)
        
        # Check if 'question' column exists
        if 'question' in df.columns:
            questions = df['question'].tolist()
            print(f"✅ Loaded {len(questions)} questions from {csv_file}")
            return questions
        else:
            print(f"❌ No 'question' column found in {csv_file}")
            print(f"Available columns: {list(df.columns)}")
            return []
            
    except FileNotFoundError:
        print(f"❌ CSV file '{csv_file}' not found")
        
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return []

# Load questions from CSV
test_questions = load_questions_from_csv()

if not test_questions:
    print("❌ No questions loaded.")

# Test each question and collect results
results_data = []

for i, question in enumerate(test_questions, 1):
    print(f"\n🧪 Test {i}: {question}")
    print("-" * 50)
    
    try:
        result = graph.invoke({"question": question})
        
        print(f"📊 Context retrieved: {len(result['context'])} documents")
        print(f"📈 Financial data: {result['financial_data'][:100]}...")
        print(f"💡 AI Response: {result['answer']}")
        
        # Store results for CSV export
        results_data.append({
            'question': question,
            'ai_response': result['answer'],
            'financial_data': result['financial_data'],
            'context_count': len(result['context']),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
    except Exception as e:
        print(f"❌ Error: {e}")
        results_data.append({
            'question': question,
            'ai_response': f"Error: {str(e)}",
            'financial_data': '',
            'context_count': 0,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    
    print("-" * 50)

# Save results to CSV
def save_results_to_csv(results: List[dict], filename: str = "financial_advisor_results.csv"):
    """
    Save test results to CSV file.
    """
    try:
        df = pd.DataFrame(results)
        df.to_csv(filename, index=False, encoding='utf-8')
        print(f"\n✅ Results saved to {filename}")
        print(f"📊 Total questions tested: {len(results)}")
        
        # Show summary
        successful_tests = len([r for r in results if not r['ai_response'].startswith('Error')])
        print(f"✅ Successful responses: {successful_tests}")
        print(f"❌ Failed responses: {len(results) - successful_tests}")
        
    except Exception as e:
        print(f"❌ Error saving results: {e}")

# Save results
save_results_to_csv(results_data)

print("="*60)
print("FINANCIAL ADVISOR RAG SYSTEM TESTING COMPLETE")
print("="*60)

# =============================================================================
# Graph Visualization
# =============================================================================
# Save the graph image to a file
print("Saving financial advisor graph visualization...")
try:
    # Generate the graph image
    graph_image = graph.get_graph().draw_mermaid_png()
    
    # Save to file
    with open("rag_graph.png", "wb") as f:
        f.write(graph_image)
    
    print("✅ Financial Advisor Graph saved as 'financial_advisor_rag_graph.png'")
    print("📁 You can find the image in your current directory")
    
except Exception as e:
    print(f"❌ Error saving graph: {e}")
    print("💡 Make sure you have the required dependencies installed")

# =============================================================================
# Usage Instructions
# =============================================================================
print("\n" + "="*60)
print("FINANCIAL ADVISOR USAGE INSTRUCTIONS")
print("="*60)
print("""
🎯 VERWENDUNG:
1. Starten Sie das System: python financial_advisor_rag.py
2. Das System lädt automatisch Fragen aus catalog.csv
3. Jede Frage wird mit dem RAG-System beantwortet
4. Ergebnisse werden in financial_advisor_results.csv gespeichert

📁 CSV-DATEIEN:
- catalog.csv: Eingabefragen (Spalte: 'question')
- financial_advisor_results.csv: Ausgabeergebnisse
  - question: Ursprungsfrage
  - ai_response: KI-Antwort
  - financial_data: Aktuelle Finanzdaten
  - context_count: Anzahl geladener Dokumente
  - timestamp: Zeitstempel

🔒 COMPLIANCE-FEATURES:
- Automatische KI-Kennzeichnung
- Risiko-Hinweise bei jeder Antwort
- Empfehlung professioneller Beratung
- Keine spezifischen Anlageempfehlungen
- Transparente Datenquellen

📊 DATENQUELLEN:
- Investopedia (Finanzwissen)
- Yahoo Finance (Aktuelle Kurse)
- Portfolio-Beispiele
- Marktübersichten

💡 BEISPIEL-FRAGEN (catalog.csv):
- "Was ist der beste Weg für einen Anfänger zu investieren?"
- "Wie funktioniert ein ETF und welche Risiken gibt es?"
- "Sollte ich in Apple Aktien investieren?"
- "Wie kann ich mein Portfolio diversifizieren?"
- "Was sind die aktuellen Trends im Technologie-Sektor?"

🔄 WORKFLOW:
1. Fragen aus catalog.csv laden
2. RAG-System für jede Frage ausführen
3. Finanzdaten automatisch integrieren
4. EU AI Act konforme Antworten generieren
5. Ergebnisse in CSV speichern
6. Zusammenfassung anzeigen
""")