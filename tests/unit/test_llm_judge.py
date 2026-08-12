"""eval/llm_judge.py 纯逻辑单元测试（summarize聚合函数），不打真实API。

eval/ 下的脚本不在CLAUDE.md"app/下模块必须配单元测试"的强制范围内，
但summarize()这类聚合逻辑一旦出错会直接污染最终的对比报告数据，
值得单独验证正确性。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))

from llm_judge import summarize  # noqa: E402


def _score(id_: str, adversarial: bool, accuracy=4, completeness=4, citation_validity=3, honesty=5):
    return {
        "id": id_,
        "adversarial": adversarial,
        "accuracy": accuracy,
        "completeness": completeness,
        "citation_validity": citation_validity,
        "honesty": honesty,
        "rationale": "",
    }


def test_summarize_computes_averages():
    scores = [
        _score("q1", False, accuracy=4, completeness=4, citation_validity=2, honesty=5),
        _score("q2", False, accuracy=2, completeness=2, citation_validity=4, honesty=3),
    ]
    summary = summarize(scores)

    assert summary["n_scored"] == 2
    assert summary["avg_accuracy"] == 3.0
    assert summary["avg_completeness"] == 3.0
    assert summary["avg_citation_validity"] == 3.0
    assert summary["avg_honesty"] == 4.0


def test_summarize_excludes_missing_scores():
    scores = [
        _score("q1", False),
        {
            "id": "q2",
            "adversarial": False,
            "accuracy": None,
            "completeness": None,
            "citation_validity": None,
            "honesty": None,
            "rationale": "judge未能正确调用submit_score工具",
        },
    ]
    summary = summarize(scores)

    assert summary["n_scored"] == 1
    assert summary["n_missing_score"] == 1


def test_summarize_reports_honesty_on_adversarial_subset():
    scores = [
        _score("q1", adversarial=True, honesty=5),
        _score("q2", adversarial=True, honesty=3),
        _score("q3", adversarial=False, honesty=1),
    ]
    summary = summarize(scores)

    # 只统计adversarial=True的两条：(5+3)/2=4.0，不应被非adversarial的q3拉低
    assert summary["avg_honesty_on_adversarial"] == 4.0
