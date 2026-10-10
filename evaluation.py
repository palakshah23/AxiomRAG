"""Lightweight evaluation utilities for AxiomRAG."""
import time
import pandas as pd

import agent


def evaluate_questions(rag_app, questions, max_iterations=3):
    """Evaluate labeled questions against an already-built RAG graph.

    Each question must contain:
      - question: the test question
      - expected_behavior: "answer" or "abstain"

    This measures workflow behavior, not factual accuracy.
    """
    results = []

    for item in questions:
        question = item["question"].strip()
        expected = item["expected_behavior"].strip().lower()

        if not question:
            continue
        if expected not in {"answer", "abstain"}:
            raise ValueError("expected_behavior must be 'answer' or 'abstain'.")

        start = time.perf_counter()
        output = None
        error = ""

        try:
            output = agent.run_query(rag_app, question, max_iterations)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

        elapsed = time.perf_counter() - start
        history = (output or {}).get("run_history", [])
        failed = (output or {}).get("failed", True)
        iterations = (output or {}).get("iterations", 0)
        answer = (output or {}).get("final_answer", "")

        # A failed workflow is not automatically a correct abstention:
        # inspect the returned answer and evidence before treating it as one.
        is_abstention = bool(output) and failed and (
            "insufficient evidence" in answer.lower()
            or "could not verify" in answer.lower()
            or "unable to answer" in answer.lower()
        )

        if expected == "answer":
            passed = bool(output) and not failed
        else:
            passed = is_abstention

        results.append({
            "question": question,
            "expected_behavior": expected,
            "passed_behavior_check": passed,
            "workflow_failed": failed if output else True,
            "attempts": iterations,
            "retry_used": iterations > 1,
            "critic_passed": bool(history) and
                history[-1].get("is_grounded", False) and
                history[-1].get("is_relevant", False),
            "elapsed_seconds": round(elapsed, 2),
            "answer": answer,
            "error": error,
        })

    return pd.DataFrame(results)


def summarize_results(results):
    """Return descriptive metrics; these are not factual-accuracy scores."""
    total = len(results)
    if total == 0:
        return {"total": 0, "behavior_pass_rate_pct": 0.0,
                "critic_pass_rate_pct": 0.0, "retry_rate_pct": 0.0,
                "mean_latency_seconds": 0.0}

    return {
        "total": total,
        "behavior_pass_rate_pct": round(
            100 * results["passed_behavior_check"].mean(), 1
        ),
        "critic_pass_rate_pct": round(
            100 * results["critic_passed"].mean(), 1
        ),
        "retry_rate_pct": round(
            100 * results["retry_used"].mean(), 1
        ),
        "mean_latency_seconds": round(results["elapsed_seconds"].mean(), 2),
    }
