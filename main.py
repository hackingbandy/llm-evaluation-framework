# main.py
# --- Main FastAPI Application ---

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pandas as pd
import io
import os
import json
from typing import List, Dict, Any

# Import modules from the framework
from llm_interaction import get_llm_response
from analysis import analyze_response

# --- Application Setup ---

app = FastAPI(
    title="LLM Evaluation Framework",
    description="Upload a CSV or JSON file with questions, get LLM responses, and analyze them.",
    version="1.0.0",
)

# --- Static Files ---
# Create a 'static' directory in the same folder as main.py for your HTML file
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- In-memory Storage for Data and Results ---
# In a production environment, you would use a proper database.
data_store: Dict[str, pd.DataFrame] = {}
results_store: Dict[str, List[Dict[str, Any]]] = {}
conversation_histories: Dict[str, List[Dict[str, str]]] = {}

# --- Pydantic Models for API Requests ---
class Query(BaseModel):
    session_id: str
    question: str
    row_index: int

class FollowUpQuery(BaseModel):
    session_id: str
    question: str
    row_index: int

# --- Helper Functions ---

async def parse_file(file: UploadFile) -> pd.DataFrame:
    """Parses the uploaded CSV or JSON file into a pandas DataFrame."""
    try:
        content = await file.read()
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        elif file.filename.endswith('.json'):
            df = pd.read_json(io.StringIO(content.decode('utf-8')))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Please upload a CSV or JSON file.")
        
        if 'question' not in df.columns:
            raise HTTPException(status_code=400, detail="The uploaded file must contain a 'question' column.")
            
        return df
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error parsing file: {e}")


# --- FastAPI Endpoints ---

@app.get("/", response_class=HTMLResponse)
async def get_index():
    """Serves the main HTML page for file upload and interaction."""
    # Assumes you have an index.html file in the 'static' directory
    with open("static/index.html") as f:
        return HTMLResponse(content=f.read(), status_code=200)

@app.post("/upload")
async def upload_data(file: UploadFile = File(...)):
    """
    Handles file uploads, parses the data, and prepares it for processing.
    """
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded.")
    
    session_id = os.urandom(16).hex()
    df = await parse_file(file)
    data_store[session_id] = df
    results_store[session_id] = []
    
    # Initialize conversation histories for each question in the file
    for i in range(len(df)):
        conversation_histories[f"{session_id}_{i}"] = []

    return {"session_id": session_id, "data": df.to_dict(orient='records')}


@app.post("/ask", response_class=JSONResponse)
async def ask_question(query: Query):
    """
    Takes a question from the uploaded data and gets a response from the LLM.
    """
    if query.session_id not in data_store:
        raise HTTPException(status_code=404, detail="Session not found.")
    
    history_key = f"{query.session_id}_{query.row_index}"
    history = conversation_histories.get(history_key, [])

    llm_response_text = get_llm_response(query.question, history)
    
    # Update conversation history
    history.append({"role": "user", "content": query.question})
    history.append({"role": "assistant", "content": llm_response_text})
    conversation_histories[history_key] = history

    # Analyze the response
    df = data_store[query.session_id]
    ground_truth = df.iloc[query.row_index].get('ground_truth', None) # Optional ground_truth
    
    analysis_results = analyze_response(llm_response_text, ground_truth)

    result = {
        "question": query.question,
        "llm_response": llm_response_text,
        "analysis": analysis_results,
        "row_index": query.row_index,
    }
    
    # Store the result for this session
    # Avoid duplicates if the user asks the same question multiple times
    existing_indices = [res['row_index'] for res in results_store[query.session_id]]
    if query.row_index not in existing_indices:
        results_store[query.session_id].append(result)

    return result

@app.post("/followup", response_class=JSONResponse)
async def ask_followup_question(query: FollowUpQuery):
    """
    Handles follow-up questions for a specific conversation thread.
    """
    if query.session_id not in data_store:
        raise HTTPException(status_code=404, detail="Session not found.")

    history_key = f"{query.session_id}_{query.row_index}"
    if history_key not in conversation_histories:
        raise HTTPException(status_code=404, detail="Conversation history not found for this question.")

    history = conversation_histories[history_key]
    
    llm_response_text = get_llm_response(query.question, history)

    # Update conversation history
    history.append({"role": "user", "content": query.question})
    history.append({"role": "assistant", "content": llm_response_text})
    conversation_histories[history_key] = history

    # You might want a different or simpler analysis for follow-ups
    analysis_results = analyze_response(llm_response_text) 

    return {
        "question": query.question,
        "llm_response": llm_response_text,
        "analysis": analysis_results,
        "row_index": query.row_index
    }

@app.get("/results/{session_id}")
async def get_results(session_id: str):
    """
    Retrieves all the stored results for a given session.
    """
    if session_id not in results_store:
        raise HTTPException(status_code=404, detail="Session not found.")
    
    return JSONResponse(content=results_store[session_id])

# To run this app:
# 1. Make sure you have all packages from requirements.txt installed.
# 2. Run `uvicorn main:app --reload` in your terminal.
# 3. Open your browser to http://127.0.0.1:8000


# .env
# IMPORTANT: Create this file in the same directory as your main.py
# and replace "your_api_key_here" with your actual OpenAI API key.

# OPENAI_API_KEY="your_api_key_here"