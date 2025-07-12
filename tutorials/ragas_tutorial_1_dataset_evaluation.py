# test data
import os
import asyncio
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

openai_api_key = os.getenv("OPENAI_API_KEY")

evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
evaluator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

from ragas import SingleTurnSample
from ragas.metrics import AspectCritic, Faithfulness, AnswerRelevancy

async def main():
    test_data = {
        "user_input": "summarise given text\nThe company reported an 8% rise in Q3 2024, driven by strong performance in the Asian market. Sales in this region have significantly contributed to the overall growth. Analysts attribute this success to strategic marketing and product localization. The positive trend in the Asian market is expected to continue into the next quarter.",
        "response": "The company experienced an 8% increase in Q3 2024, largely due to effective marketing strategies and product adaptation, with expectations of continued growth in the coming quarter.",
    }

    metric = AspectCritic(name="summary_accuracy", llm=evaluator_llm, definition="Verify if the summary is accurate.")
    test_data = SingleTurnSample(**test_data)
    result = await metric.single_turn_ascore(test_data)
    print(f"Score: {result}")

if __name__ == "__main__":
    asyncio.run(main())

# =============== Dataset ===============
from datasets import load_dataset
from ragas import EvaluationDataset
eval_dataset = load_dataset("explodinggradients/earning_report_summary",split="train")
eval_dataset = EvaluationDataset.from_hf_dataset(eval_dataset)

metric = AspectCritic(name="summary_accuracy", llm=evaluator_llm, definition="Verify if the summary is accurate.")
print("Features in dataset:", eval_dataset.features())
print("Total samples in dataset:", len(eval_dataset))

from ragas import evaluate

results = evaluate(eval_dataset, metrics=[metric])
print(results)

print(results.to_pandas())