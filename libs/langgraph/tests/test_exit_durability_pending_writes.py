"""Regression test for exit-durability writes from late-finishing siblings."""

import threading
import time
from collections import Counter
from typing import TypedDict

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Durability


class _State(TypedDict, total=False):
    a: int
    b: int


@pytest.mark.parametrize("durability", ["sync", "async", "exit"])
def test_completed_sibling_is_not_replayed_after_failure(
    durability: Durability,
) -> None:
    """A successful sibling that finishes during shutdown must be resumable."""
    runs: Counter[str] = Counter()
    slow_started = threading.Event()
    failure_starting = threading.Event()

    def flaky(state: _State) -> dict:
        runs["flaky"] += 1
        assert slow_started.wait(timeout=5)
        if runs["flaky"] == 1:
            failure_starting.set()
            raise RuntimeError("transient failure")
        return {"a": 1}

    def slow(state: _State) -> dict:
        runs["slow"] += 1
        slow_started.set()
        assert failure_starting.wait(timeout=5)
        time.sleep(0.1)
        return {"b": 1}

    saver = InMemorySaver()
    builder = StateGraph(_State)
    builder.add_node("flaky", flaky)
    builder.add_node("slow", slow)
    builder.add_edge(START, "flaky")
    builder.add_edge(START, "slow")
    builder.add_edge(["flaky", "slow"], END)
    graph = builder.compile(checkpointer=saver)
    config = {
        "configurable": {"thread_id": f"late-write-{durability}"},
        "max_concurrency": 2,
    }

    with pytest.raises(RuntimeError, match="transient failure"):
        graph.invoke({}, config, durability=durability)

    saved = saver.get_tuple(config)
    assert saved is not None
    assert ("b", 1) in [
        (channel, value) for _, channel, value in saved.pending_writes
    ]
    assert runs == {"flaky": 1, "slow": 1}

    assert graph.invoke(None, config, durability=durability) == {"a": 1, "b": 1}
    assert runs == {"flaky": 2, "slow": 1}
