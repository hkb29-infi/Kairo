import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

from rag import RAGService, RetrievalMode

METRIC_NAMES = ("context_precision", "faithfulness", "answer_relevancy")


@dataclass(frozen=True)
class EvalCase:
    question: str
    reference: str


def load_golden_dataset(
    path: str | Path, minimum_samples: int = 50
) -> list[EvalCase]:
    if minimum_samples < 1:
        raise ValueError("minimum_samples must be at least 1.")
    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Evaluation dataset not found: {dataset_path}")

    cases: list[EvalCase] = []
    with dataset_path.open(encoding="utf-8") as dataset_file:
        for line_number, line in enumerate(dataset_file, start=1):
            if not line.strip():
                continue
            try:
                record: Any = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {dataset_path}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"Line {line_number} must contain a JSON object."
                )
            question = record.get("question")
            reference = record.get("reference")
            if not isinstance(question, str) or not question.strip():
                raise ValueError(
                    f"Line {line_number} must include a non-empty question."
                )
            if not isinstance(reference, str) or not reference.strip():
                raise ValueError(
                    f"Line {line_number} must include a non-empty reference."
                )
            cases.append(
                EvalCase(question=question.strip(), reference=reference.strip())
            )

    if len(cases) < minimum_samples:
        raise ValueError(
            f"Evaluation requires at least {minimum_samples} examples; "
            f"found {len(cases)}."
        )
    return cases


def _evaluate_variant(
    cases: list[EvalCase],
    service: RAGService,
    retrieval_mode: RetrievalMode,
    judge_llm: Any,
    judge_embeddings: Any,
) -> tuple[dict[str, float], list[float], list[dict[str, float]]]:
    from ragas import EvaluationDataset, evaluate
    from ragas.metrics import (
        Faithfulness,
        LLMContextPrecisionWithReference,
        ResponseRelevancy,
    )

    records: list[dict[str, Any]] = []
    latencies: list[float] = []
    for case in cases:
        result = service.ask(case.question, retrieval_mode=retrieval_mode)
        records.append(
            {
                "user_input": case.question,
                "response": result.answer,
                "retrieved_contexts": result.contexts,
                "reference": case.reference,
            }
        )
        latencies.append(result.latency_ms)

    metrics = [
        LLMContextPrecisionWithReference(name="context_precision"),
        Faithfulness(name="faithfulness"),
        ResponseRelevancy(name="answer_relevancy"),
    ]
    evaluation_result = evaluate(
        dataset=EvaluationDataset.from_list(records),
        metrics=metrics,
        llm=judge_llm,
        embeddings=judge_embeddings,
        raise_exceptions=True,
        show_progress=True,
    )
    scores = evaluation_result.to_pandas()
    if len(scores) != len(cases):
        raise RuntimeError(
            f"RAGAS returned {len(scores)} scores for {len(cases)} examples."
        )

    means: dict[str, float] = {}
    per_example: list[dict[str, float]] = [dict() for _ in cases]
    for name in METRIC_NAMES:
        if name not in scores:
            raise RuntimeError(f"RAGAS did not return the '{name}' metric.")
        values = [float(value) for value in scores[name]]
        if not all(math.isfinite(value) for value in values):
            raise RuntimeError(f"RAGAS returned invalid '{name}' scores.")
        means[name] = fmean(values)
        for index, value in enumerate(values):
            per_example[index][name] = value
    return means, latencies, per_example


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _variant_summary(
    metrics: dict[str, float], latencies: list[float]
) -> dict[str, Any]:
    if not latencies or not all(math.isfinite(value) for value in latencies):
        raise ValueError("Latency measurements must be finite and non-empty.")
    return {
        "metrics": metrics,
        "composite_score": fmean(metrics.values()),
        "latency_ms": {
            "mean": fmean(latencies),
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
        },
    }


def run_comparison(
    dataset_path: str | Path,
    cases: list[EvalCase],
    service: RAGService,
    judge_llm: Any,
    judge_embeddings: Any,
    judge_model: str,
    generation_model: str,
) -> dict[str, Any]:
    import ragas

    dataset_file = Path(dataset_path)
    baseline_metrics, baseline_latencies, baseline_samples = _evaluate_variant(
        cases, service, "similarity", judge_llm, judge_embeddings
    )
    mmr_metrics, mmr_latencies, mmr_samples = _evaluate_variant(
        cases, service, "mmr", judge_llm, judge_embeddings
    )
    baseline = _variant_summary(baseline_metrics, baseline_latencies)
    mmr = _variant_summary(mmr_metrics, mmr_latencies)
    baseline_score = baseline["composite_score"]
    mmr_score = mmr["composite_score"]
    relative_improvement = (
        (mmr_score - baseline_score) / baseline_score * 100
        if baseline_score > 0
        else None
    )
    return {
        "dataset_sha256": hashlib.sha256(dataset_file.read_bytes()).hexdigest(),
        "sample_count": len(cases),
        "generation_model": generation_model,
        "judge_model": judge_model,
        "generation_temperature": 0,
        "retrieval_k": 3,
        "ragas_version": ragas.__version__,
        "composite_method": "arithmetic mean of context precision, faithfulness, and answer relevancy",
        "variants": {"similarity": baseline, "mmr": mmr},
        "mmr_vs_similarity": {
            "absolute_improvement_points": mmr_score - baseline_score,
            "relative_improvement_percent": relative_improvement,
        },
        "per_example": [
            {
                "index": index,
                "similarity": {
                    "metrics": baseline_samples[index],
                    "latency_ms": baseline_latencies[index],
                },
                "mmr": {
                    "metrics": mmr_samples[index],
                    "latency_ms": mmr_latencies[index],
                },
            }
            for index in range(len(cases))
        ],
    }
