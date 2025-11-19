import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)

from evaluation import load_golden_dataset, run_comparison
from rag import create_rag_service


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare similarity and MMR retrieval with RAGAS LLM judging."
    )
    parser.add_argument("--dataset", required=True, help="Golden JSONL dataset.")
    parser.add_argument(
        "--output",
        default="evaluation/results.json",
        help="Path to write the comparison report.",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=50,
        help="Minimum dataset size (default: 50).",
    )
    args = parser.parse_args()

    load_dotenv()
    cases = load_golden_dataset(args.dataset, minimum_samples=args.min_samples)
    api_key = os.environ["GOOGLE_API_KEY"]
    judge_model = os.getenv("GEMINI_JUDGE_MODEL", "gemini-2.5-flash")
    judge_llm = ChatGoogleGenerativeAI(
        model=judge_model,
        temperature=0,
        api_key=api_key,
    )
    judge_embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key,
    )
    report = run_comparison(
        dataset_path=args.dataset,
        cases=cases,
        service=create_rag_service(),
        judge_llm=judge_llm,
        judge_embeddings=judge_embeddings,
        judge_model=judge_model,
        generation_model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, allow_nan=False))
    print(f"Report written to {output_path}")


if __name__ == "__main__":
    main()
