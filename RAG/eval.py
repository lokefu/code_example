import os
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision
)
# If using OpenAI as the Judge (Recommended for accuracy)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# 1. Setup the "Judge" LLM
# Ragas needs an LLM to act as the evaluator to decompose statements and check logic.
os.environ["OPENAI_API_KEY"] = "your-api-key"
evaluator_llm = ChatOpenAI(model="gpt-4o") 

def run_benchmark(rag_pipeline, test_questions, ground_truths):
    """
    rag_pipeline: Your LangGraph / Mistral inference function
    test_questions: List of user queries
    ground_truths: List of expected answers (from your data.json)
    """
    
    results_collector = []

    print(f"🚀 Starting evaluation on {len(test_questions)} queries...")

    for query, truth in zip(test_questions, ground_truths):
        # Execution: Get response from your RAG app
        # This assumes your pipeline returns the answer and the retrieved documents
        response = rag_pipeline.invoke({"question": query})
        
        # Extract data from your LangGraph state
        generated_answer = response.get("generation")
        # Ragas expects contexts as a list of strings
        retrieved_contexts = [doc.page_content for doc in response.get("documents", [])]

        results_collector.append({
            "question": query,
            "answer": generated_answer,
            "contexts": retrieved_contexts,
            "ground_truth": truth
        })

    # 2. Convert to Dataset format required by Ragas
    dataset = Dataset.from_list(results_collector)

    # 3. Perform Evaluation
    print("📊 Calculating RAGAS metrics...")
    score = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_recall,
            context_precision
        ],
        llm=evaluator_llm,
        embeddings=OpenAIEmbeddings()
    )

    # 4. Process and Save Results
    df = score.to_pandas()
    df.to_csv("rag_evaluation_report.csv", index=False)
    
    print("\n✅ Evaluation Complete!")
    print("-" * 30)
    print(score)
    print("-" * 30)
    
    return df

if __name__ == "__main__":
    # Example Test Set based on your biodiversity data
    test_queries = [
        "What is the scientific name of the Malayan Colugo?",
        "Where can I find the Sunda Pangolin in Singapore?",
        "Describe the diet of the Oriental Pied Hornbill."
    ]
    
    expected_answers = [
        "Galeopterus variegatus",
        "Central Nature Reserve and Pulau Ubin",
        "Mainly fruit (figs), but also small insects and reptiles."
    ]

    # Import your actual graph/pipeline here
    # from your_main_script import app 
    
    # report = run_benchmark(app, test_queries, expected_answers)
