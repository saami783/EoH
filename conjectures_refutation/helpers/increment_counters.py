import json
from pathlib import Path
from filelock import FileLock

JSON_FILE = Path("counter.json")
lock = FileLock("counter.json.lock")  # type: ignore


def _increment(key: str):
    with lock.acquire(timeout=10):
        data = json.loads(JSON_FILE.read_text(encoding="utf-8"))

        data[key] = data.get(key, 0) + 1

        JSON_FILE.write_text(json.dumps(data, indent=4), encoding="utf-8")


def increment_counter_graph_generation():
    _increment("total_graphs_generated")


def increment_counter_mutations():
    _increment("total_graphs_muted")

