# =============================================================================
# LLM Evaluation Framework - RAG (Retrieval-Augmented Generation) Tutorial
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
# Initialize LLM
llm = init_chat_model("gpt-4o-mini", model_provider="openai")

# Initialize Embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Initialize Vector Store
vector_store = InMemoryVectorStore(embeddings)

# =============================================================================
# Data Loading & Processing
# =============================================================================
# Load and chunk contents of the blog
loader = WebBaseLoader(
    web_paths=("https://www.codecademy.com/resources/blog/6-popular-use-cases-for-building-ai-skills/",)
    # Removed CSS filtering to capture all content
)
docs = loader.load()

# Verify document loaded correctly
assert len(docs) == 1
print(f"Total characters: {len(docs[0].page_content)}")

# Display first 500 characters for verification
print("First 500 characters:")
print(docs[0].page_content[:500])
print("-" * 50)

# Split document into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # chunk size (characters)
    chunk_overlap=200,    # chunk overlap (characters)
    add_start_index=True, # track index in original document
)
all_splits = text_splitter.split_documents(docs)

print(f"Split blog post into {len(all_splits)} sub-documents.")

# Index chunks in vector store
document_ids = vector_store.add_documents(documents=all_splits)
print(f"First 3 document IDs: {document_ids[:3]}")

# =============================================================================
# Prompt Configuration
# =============================================================================
# Load RAG prompt from LangChain Hub
# Note: For non-US LangSmith endpoints, you may need to specify
# api_url="https://api.smith.langchain.com" in hub.pull.
prompt = hub.pull("rlm/rag-prompt")

# Test the prompt
example_messages = prompt.invoke(
    {"context": "(context goes here)", "question": "(question goes here)"}
).to_messages()

assert len(example_messages) == 1
print("Example prompt content:")
print(example_messages[0].content)
print("-" * 50)

# =============================================================================
# Type Definitions
# =============================================================================
class State(TypedDict):
    """Application state for the RAG graph."""
    question: str
    context: List[Document]
    answer: str

# =============================================================================
# Graph Functions
# =============================================================================
def retrieve(state: State) -> dict:
    """
    Retrieve relevant documents based on the question.
    """
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}

def generate(state: State) -> dict:
    """
    Generate answer based on retrieved context using the RAG prompt.
    """
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

# =============================================================================
# Graph Construction
# =============================================================================
# Build the RAG graph
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

# =============================================================================
# Custom Prompt Definition (Alternative)
# =============================================================================
# Define custom RAG prompt as alternative to hub prompt
custom_template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.
Always say "thanks for asking!" at the end of the answer.

{context}

Question: {question}

Helpful Answer:"""
custom_rag_prompt = PromptTemplate.from_template(custom_template)

# =============================================================================
# Test Execution
# =============================================================================
print("="*50)
print("TESTING THE RAG SYSTEM")
print("="*50)

# Test the graph with a simple question
print("Testing with question: 'What is the ai-maturiy level of Siemens AG?'")
result = graph.invoke({"question": "What is the ai-maturiy level of Siemens AG?"})

print(f'\nContext retrieved: {len(result["context"])} documents')
print(f'Answer: {result["answer"]}')
print("-" * 50)

# Stream the graph execution for debugging
# print("\nStreaming execution for debugging:")
# for step in graph.stream(
#     {"question": "What is Task Decomposition?"}, stream_mode="updates"
# ):
#     print(f"Step: {step}")
#     print("-" * 30)

print("="*50)
print("RAG SYSTEM TESTING COMPLETE")
print("="*50)

# =============================================================================
# Graph Visualization
# =============================================================================
# Save the graph image to a file
print("Saving graph visualization...")
try:
    # Generate the graph image
    graph_image = graph.get_graph().draw_mermaid_png()
    
    # Save to file
    with open("rag_graph.png", "wb") as f:
        f.write(graph_image)
    
    print("✅ Graph saved as 'rag_graph.png'")
    print("📁 You can find the image in your current directory")
    
except Exception as e:
    print(f"❌ Error saving graph: {e}")
    print("💡 Make sure you have the required dependencies installed")

# Optional: Display the graph (requires IPython)
# Uncomment the following lines if you want to visualize the graph:
# from IPython.display import Image, display
# display(Image(graph.get_graph().draw_mermaid_png()))