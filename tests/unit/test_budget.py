"""app/infra/budget.py 单元测试：纯函数，不涉及LLM/网络。"""
import time

from app.infra.budget import BudgetLimits, add_usage, elapsed_seconds, is_exhausted, new_budget


def test_new_budget_starts_at_zero():
    budget = new_budget()
    assert budget["tokens_used"] == 0
    assert budget["tool_calls_used"] == 0


def test_add_usage_accumulates_without_mutating_original():
    budget = new_budget()
    updated = add_usage(budget, tokens=100, tool_calls=2)

    assert updated["tokens_used"] == 100
    assert updated["tool_calls_used"] == 2
    assert budget["tokens_used"] == 0  # 原对象不应被原地修改

    updated2 = add_usage(updated, tokens=50, tool_calls=1)
    assert updated2["tokens_used"] == 150
    assert updated2["tool_calls_used"] == 3


def test_is_exhausted_false_when_under_all_limits():
    budget = new_budget()
    limits = BudgetLimits(max_tokens=1000, max_tool_calls=10, max_wall_clock_seconds=60)
    assert is_exhausted(budget, limits) is False


def test_is_exhausted_true_when_tokens_exceed_limit():
    budget = add_usage(new_budget(), tokens=1000)
    limits = BudgetLimits(max_tokens=100, max_tool_calls=10, max_wall_clock_seconds=60)
    assert is_exhausted(budget, limits) is True


def test_is_exhausted_true_when_tool_calls_exceed_limit():
    budget = add_usage(new_budget(), tool_calls=20)
    limits = BudgetLimits(max_tokens=10**9, max_tool_calls=5, max_wall_clock_seconds=60)
    assert is_exhausted(budget, limits) is True


def test_is_exhausted_true_when_wall_clock_exceeds_limit():
    budget = {"tokens_used": 0, "tool_calls_used": 0, "wall_clock_start": time.monotonic() - 100}
    limits = BudgetLimits(max_tokens=10**9, max_tool_calls=10**9, max_wall_clock_seconds=1.0)
    assert is_exhausted(budget, limits) is True


def test_elapsed_seconds_grows_over_time():
    budget = new_budget()
    time.sleep(0.05)
    assert elapsed_seconds(budget) >= 0.05
