# =============================================================================
# LLM Evaluation Framework - Query Analysis with LangGraph
# =============================================================================

# =============================================================================
# Environment Setup & Imports
# =============================================================================
import os
from typing import List, Literal
from typing_extensions import Annotated, TypedDict
from dotenv import load_dotenv

# LangChain imports
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import START, StateGraph

# =============================================================================
# Environment Configuration
# =============================================================================
# Load environment variables
load_dotenv()

# Get API keys and configuration
openai_api_key = os.getenv("OPENAI_API_KEY")
user_agent = os.getenv("USER_AGENT")

# Set USER_AGENT if not present
if not user_agent:
    os.environ['USER_AGENT'] = 'LLM-Evaluation-Framework/1.0'

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
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load document
loader = WebBaseLoader(
    "https://lilianweng.github.io/posts/2023-06-23-agent/"
)
docs = loader.load()

# Verify document loaded correctly
assert len(docs) == 1
print(f"Total characters: {len(docs[0].page_content)}")

# Split document into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

# Add documents to vector store
_ = vector_store.add_documents(all_splits)

# =============================================================================
# Metadata Management
# =============================================================================
# Add metadata to chunks (beginning, middle, end sections)
total_documents = len(all_splits)
third = total_documents // 3

for i, document in enumerate(all_splits):
    if i < third:
        document.metadata["section"] = "beginning"
    elif i < 2 * third:
        document.metadata["section"] = "middle"
    else:
        document.metadata["section"] = "end"

# Verify metadata assignment
print(f"First document metadata: {all_splits[0].metadata}")

# =============================================================================
# Type Definitions
# =============================================================================
class Search(TypedDict):
    """Search query with section specification."""
    query: Annotated[str, "Search query to run."]
    section: Annotated[
        Literal["beginning", "middle", "end"],
        "Section to query.",
    ]

class State(TypedDict):
    """Application state for the graph."""
    question: str
    query: Search
    context: List[Document]
    answer: str

# =============================================================================
# Graph Functions
# =============================================================================
def analyze_query(state: State) -> dict:
    """
    Analyze the user question and extract structured search parameters.
    """
    structured_llm = llm.with_structured_output(Search)
    query = structured_llm.invoke(state["question"])
    return {"query": query}

def retrieve(state: State) -> dict:
    """
    Retrieve relevant documents based on the structured query.
    """
    query = state["query"]
    retrieved_docs = vector_store.similarity_search(
        query["query"],
        filter=lambda doc: doc.metadata.get("section") == query["section"],
    )
    return {"context": retrieved_docs}

def generate(state: State) -> dict:
    """
    Generate answer based on retrieved context.
    """
    # Define RAG prompt
    template = """Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Use three sentences maximum and keep the answer as concise as possible.
    Always say "thanks for asking!" at the end of the answer.

    {context}

    Question: {question}

    Helpful Answer:"""
    
    rag_prompt = ChatPromptTemplate.from_template(template)
    
    # Generate response
    chain = rag_prompt | llm
    answer = chain.invoke({"context": state["context"], "question": state["question"]})
    
    return {"answer": answer.content}

# =============================================================================
# Graph Construction & Execution
# =============================================================================
# Build the graph
graph_builder = StateGraph(State).add_sequence([analyze_query, retrieve, generate])
graph_builder.add_edge(START, "analyze_query")
graph = graph_builder.compile()

# =============================================================================
# Test Execution
# =============================================================================
print("="*50)
print("TESTING THE GRAPH")
print("="*50)

# Stream the graph execution
for step in graph.stream(
    {"question": "What does the end of the post say about Task Decomposition?"},
    stream_mode="updates",
):
    print(f"Step: {step}")
    print("-" * 30)