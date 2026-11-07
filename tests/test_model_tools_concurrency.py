"""Unit test verifying that _last_resolved_tool_names is isolated across concurrent threads via ContextVar."""

import threading
import pytest
import model_tools


def test_concurrent_tool_resolution_isolation():
    results = {}
    barrier = threading.Barrier(2)

    def worker(name, tools):
        model_tools.set_last_resolved_tool_names(tools)
        barrier.wait()
        results[name] = list(model_tools._last_resolved_tool_names)

    t1 = threading.Thread(target=worker, args=("worker1", ["terminal", "read_file"]))
    t2 = threading.Thread(target=worker, args=("worker2", ["web_search", "web_extract"]))

    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert results["worker1"] == ["terminal", "read_file"]
    assert results["worker2"] == ["web_search", "web_extract"]
