"""Unit tests for tiered memory store: auto-archival on overflow, cold storage search and restore."""

import json
from pathlib import Path
from tools.memory_tool_store import MemoryStore


def test_memory_auto_archival_on_overflow(tmp_path, monkeypatch):
    import tools.memory_tool as mt
    monkeypatch.setattr(mt, "get_memory_dir", lambda: tmp_path)

    # Use a limit of 60 characters and enable auto_archive_overflow
    store = MemoryStore(memory_char_limit=60, user_char_limit=60, auto_archive_overflow=True)
    store.load_from_disk()

    # Add entry 1 (25 chars)
    e1 = "Fact 1: User likes Python"
    res1 = store.add("memory", e1)
    assert res1.get("success") is True

    # Add entry 2 (25 chars) -> Total 25 + 3 + 25 = 53 chars
    e2 = "Fact 2: User uses Windows"
    res2 = store.add("memory", e2)
    assert res2.get("success") is True

    # Add entry 3 (24 chars) -> Total 53 + 3 + 24 = 80 chars (> 60 limit).
    # Oldest entry (e1) should be evicted to cold storage (memory_archive.jsonl).
    # Remaining: e2 (25) + 3 + e3 (24) = 52 chars (<= 60 limit).
    e3 = "Fact 3: User uses Hermes"
    res3 = store.add("memory", e3)
    assert res3.get("success") is True
    assert "Archived 1 oldest entry" in res3.get("message", "")

    # Verify active memory has e2 and e3
    entries = store._entries_for("memory")
    assert e1 not in entries
    assert e2 in entries
    assert e3 in entries

    # Verify cold storage has e1
    archive_file = tmp_path / "memory_archive.jsonl"
    assert archive_file.is_file()
    lines = archive_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["content"] == e1

    # Search across hot and cold
    search_res = store.search("memory", "Python")
    assert search_res["total_matches"] == 1
    assert search_res["results"][0]["tier"] == "cold"
    assert search_res["results"][0]["content"] == e1

    search_hot = store.search("memory", "Windows")
    assert search_hot["total_matches"] == 1
    assert search_hot["results"][0]["tier"] == "hot"
    assert search_hot["results"][0]["content"] == e2

    # Test restoring from cold storage back to active memory
    restore_res = store.restore("memory", "Python")
    assert restore_res.get("success") is True
    # Now e1 is restored into active memory (which in turn may archive e2 if over limit)
    assert e1 in store._entries_for("memory")
