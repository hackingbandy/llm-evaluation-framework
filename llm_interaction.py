# llm_interaction.py
# --- Module for Interacting with the Large Language Model ---

import os
from openai import OpenAI
from dotenv import load_dotenv

# --- LLM Configuration ---
# Load environment variables from .env file
load_dotenv()

# It's recommended to set your API key as an environment variable
# for security. Create a .env file in your project root with:
# OPENAI_API_KEY="your_api_key_here"
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in a .env file.")

print(f"API Key found: {api_key[:8]}...") # Print first 8 characters of API key for verification

client = OpenAI(api_key=api_key)

def get_llm_response(prompt: str, history: list = None) -> str:
    """
    Sends a prompt to the LLM and returns the response.
    Includes conversation history for context.

    Args:
        prompt: The user's question or prompt.
        history: A list of previous turns in the conversation.
                 Example: [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there!"}]

    Returns:
        The text response from the LLM.
    """
    if history is None:
        history = []

    messages = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
    messages.extend(history)
    messages.append({"role": "user", "content": prompt})
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # You can change the model here
            messages=messages
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"An error occurred with the LLM API: {e}")
        return "Error: Could not get a response from the model."

