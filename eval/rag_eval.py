#!/usr/bin/env python
"""RAG Evaluation Script - Faithfulness and Context Relevance checks."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class RAGEvaluator:
    """Lightweight RAG evaluation for faithfulness and relevance."""
    
    def __init__(self):
        self.results = []
    
    def evaluate_faithfulness(self, answer: str, contexts: list[str]) -> float:
        """Check if answer is grounded in contexts (simple heuristic)."""
        if not contexts or not answer:
            return 0.0
        # Count context words in answer
        context_words = set()
        for ctx in contexts:
            context_words.update(ctx.lower().split())
        answer_words = answer.lower().split()
        if not answer_words:
            return 0.0
        grounded = sum(1 for w in answer_words if w in context_words)
        return min(1.0, grounded / len(answer_words) * 2)  # scaled
    
    def evaluate_context_relevance(self, question: str, contexts: list[str]) -> float:
        """Check if contexts are relevant to question."""
        if not contexts or not question:
            return 0.0
        q_words = set(question.lower().split())
        scores = []
        for ctx in contexts:
            c_words = set(ctx.lower().split())
            overlap = len(q_words & c_words) / max(len(q_words), 1)
            scores.append(overlap)
        return sum(scores) / len(scores)
    
    def evaluate(self, test_cases: list[dict[str, Any]]) -> dict[str, Any]:
        """Run evaluation on test cases."""
        total_faith = 0.0
        total_rel = 0.0
        for case in test_cases:
            faith = self.evaluate_faithfulness(case["answer"], case["contexts"])
            rel = self.evaluate_context_relevance(case["question"], case["contexts"])
            total_faith += faith
            total_rel += rel
            self.results.append({
                "question": case["question"][:50],
                "faithfulness": round(faith, 3),
                "context_relevance": round(rel, 3),
            })
        
        n = len(test_cases)
        return {
            "num_cases": n,
            "avg_faithfulness": round(total_faith / n if n else 0, 3),
            "avg_context_relevance": round(total_rel / n if n else 0, 3),
            "results": self.results,
        }


def main():
    """Run sample evaluation."""
    evaluator = RAGEvaluator()
    
    test_cases = [
        {
            "question": "How much leave do employees get?",
            "answer": "Full-time employees receive 24 days of paid annual leave per year.",
            "contexts": ["Full-time employees receive 24 days of paid annual leave per calendar year."],
        },
        {
            "question": "What is the budget?",
            "answer": "The FY2026 operating budget totals 48.2 million USD.",
            "contexts": ["The FY2026 operating budget totals 48.2 million USD, a 6 percent increase."],
        },
    ]
    
    results = evaluator.evaluate(test_cases)
    print(json.dumps(results, indent=2))
    
    # Assertions
    assert results["avg_faithfulness"] >= 0.3, "Faithfulness too low"
    assert results["avg_context_relevance"] >= 0.2, "Relevance too low"
    print("\nAll evaluations passed")


if __name__ == "__main__":
    main()
