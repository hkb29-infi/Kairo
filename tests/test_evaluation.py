import json
from pathlib import Path

import pytest

import evaluation
from evaluation import (
    EvalCase,
    _variant_summary,
    load_golden_dataset,
)
from rag import RAGResponse


def _write_dataset(path: Path, count: int) -> None:
    path.write_text(
        "".join(
            json.dumps(
                {"question": f"Question {index}", "reference": f"Answer {index}"}
            )
            + "\n"
            for index in range(count)
        ),
        encoding="utf-8",
    )


def test_load_golden_dataset_requires_50_examples_by_default(tmp_path: Path) -> None:
    dataset = tmp_path / "golden.jsonl"
    _write_dataset(dataset, 49)

    with pytest.raises(ValueError, match="at least 50 examples"):
        load_golden_dataset(dataset)

    _write_dataset(dataset, 50)
    cases = load_golden_dataset(dataset)
    assert len(cases) == 50
    assert cases[0] == EvalCase("Question 0", "Answer 0")


def test_load_golden_dataset_reports_bad_record(tmp_path: Path) -> None:
    dataset = tmp_path / "golden.jsonl"
    dataset.write_text('{"question": "missing reference"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="non-empty reference"):
        load_golden_dataset(dataset, minimum_samples=1)


def test_variant_summary_reports_composite_and_latency_percentiles() -> None:
    summary = _variant_summary(
        {
            "context_precision": 0.8,
            "faithfulness": 0.7,
            "answer_relevancy": 0.9,
        },
        [100, 200, 300, 400],
    )

    assert summary["composite_score"] == pytest.approx(0.8)
    assert summary["latency_ms"] == {
        "mean": 250,
        "p50": 200,
        "p95": 400,
    }


def test_run_comparison_evaluates_same_cases_with_both_variants(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "golden.jsonl"
    _write_dataset(dataset, 1)
    modes: list[str] = []
    score_sets = iter(
        [
            {
                "context_precision": [0.6],
                "faithfulness": [0.6],
                "answer_relevancy": [0.6],
            },
            {
                "context_precision": [0.7],
                "faithfulness": [0.7],
                "answer_relevancy": [0.7],
            },
        ]
    )

    class FakeService:
        def ask(self, question: str, retrieval_mode: str) -> RAGResponse:
            modes.append(retrieval_mode)
            return RAGResponse("answer", ["context"], 120, retrieval_mode)

    class FakeResult:
        def __init__(self, scores: dict[str, list[float]]) -> None:
            self.scores = scores

        def to_pandas(self):
            import pandas as pd

            return pd.DataFrame(self.scores)

    def fake_evaluate(**kwargs):
        assert len(kwargs["dataset"]) == 1
        assert kwargs["raise_exceptions"] is True
        return FakeResult(next(score_sets))

    monkeypatch.setattr("ragas.evaluate", fake_evaluate)
    report = evaluation.run_comparison(
        dataset,
        [EvalCase("Question", "Reference")],
        FakeService(),
        object(),
        object(),
        "test-judge",
        "test-generator",
    )

    assert modes == ["similarity", "mmr"]
    assert report["sample_count"] == 1
    assert report["judge_model"] == "test-judge"
    assert report["generation_model"] == "test-generator"
    assert report["variants"]["similarity"]["composite_score"] == pytest.approx(0.6)
    assert report["variants"]["mmr"]["composite_score"] == pytest.approx(0.7)
    assert report["mmr_vs_similarity"]["relative_improvement_percent"] == pytest.approx(
        (0.7 - 0.6) / 0.6 * 100
    )
    assert report["per_example"][0]["mmr"]["metrics"]["faithfulness"] == 0.7
