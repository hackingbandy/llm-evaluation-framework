from rag import RAG
from typing import Dict, List



class FinancialAdvisor(RAG):
    def __init__(self):
        super().__init__()
        print("Financial Advisor initialisiert.")
FinancialAdvisor()