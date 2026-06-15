import json
import os
from concurrent.futures import ThreadPoolExecutor
import time

from clumping_factor.summary_cache import load_or_build_summary, summary_cache_identity


def identity():
    return summary_cache_identity(".", 1, "gas", "sphere", [{"path": "snap", "size": 1, "mtime_ns": 1}])


def test_corrupt_cache_is_rebuilt(tmp_path):
    calls = 0

    def build():
        nonlocal calls
        calls += 1
        return {"valid_count": 3}

    first, first_diag = load_or_build_summary("auto", tmp_path, identity(), build)
    cache_path = next(tmp_path.glob("*.json"))
    cache_path.write_text("not json")
    second, second_diag = load_or_build_summary("auto", tmp_path, identity(), build)
    assert first == second == {"valid_count": 3}
    assert calls == 2
    assert first_diag["status"] == "built"
    assert second_diag["status"] == "corrupt_rebuild"


def test_refresh_replaces_valid_cache(tmp_path):
    values = iter((1, 2))
    load_or_build_summary("auto", tmp_path, identity(), lambda: {"value": next(values)})
    refreshed, diagnostics = load_or_build_summary("refresh", tmp_path, identity(), lambda: {"value": next(values)})
    assert refreshed == {"value": 2}
    assert diagnostics["status"] == "refresh"
    document = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert document["summary"] == {"value": 2}


def test_stale_lock_is_removed(tmp_path):
    load_or_build_summary("auto", tmp_path, identity(), lambda: {"value": 1})
    cache_path = next(tmp_path.glob("*.json"))
    cache_path.unlink()
    lock_path = cache_path.with_suffix(".lock")
    lock_path.mkdir()
    old = time.time() - 100
    os.utime(lock_path, (old, old))
    summary, diagnostics = load_or_build_summary(
        "auto", tmp_path, identity(), lambda: {"value": 2}, stale_lock_seconds=1, poll_seconds=0.01
    )
    assert summary == {"value": 2}
    assert diagnostics["status"] == "stale_lock_rebuild"


def test_concurrent_callers_share_one_build(tmp_path):
    calls = 0

    def build():
        nonlocal calls
        calls += 1
        time.sleep(0.05)
        return {"value": 1}

    def run():
        return load_or_build_summary("auto", tmp_path, identity(), build, poll_seconds=0.01)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _index: run(), range(2)))
    assert calls == 1
    assert [result[0] for result in results] == [{"value": 1}, {"value": 1}]
    statuses = {result[1]["status"] for result in results}
    assert "built" in statuses
    assert statuses & {"wait_hit", "lock_hit"}
