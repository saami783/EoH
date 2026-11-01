import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Any

BASE_OUT_DIR = Path(__file__).resolve().parent.parent / "out"
PROB_METRICS_FILE = BASE_OUT_DIR / "prob_metrics.json"
COUNTER_FILE = Path("counter.json")
LOCK_FILE = Path("counter.json.lock")


@dataclass(slots=True)
class ConjectureResult:
    has_counterexample: bool
    counterexample_g6: str | None
    size_of_counter_example: int | None
    min_order: int | None
    max_order: int | None
    total_mutations: int | None
    total_mutations_of_counterexample: int | None
    score: float | None
    total_graphs_generated: int | None
    time: float | None
    total_api_requests: int | None
    seed: Optional[int] | None
    subclass: str | None
    cpus: int
    score_function_path: str
    score_function_name: str
    approx: bool
    time_limit: float
    x_value: Optional[float] | None
    y_value: Optional[float] | None


def build_result(counterexample_g6: str, size_of_counter_example: int,
                 total_mutations_of_counterexample: int, score: float,
                 x_value: Optional[float] = None, y_value: Optional[float] = None) -> None:
    BASE_OUT_DIR.mkdir(parents=True, exist_ok=True)
    prob_data = {
        "has_counterexample": True,
        "counterexample_g6": counterexample_g6,
        "size_of_counter_example": size_of_counter_example,
        "total_mutations_of_counterexample": total_mutations_of_counterexample,
        "score": score,
        "x_value": x_value,
        "y_value": y_value,
    }
    PROB_METRICS_FILE.write_text(json.dumps(prob_data, indent=4), encoding="utf-8")


def aggregate_and_log_results(params: dict[str, Any], elapsed_time: float, api_requests: int) -> ConjectureResult:
    counters = json.loads(COUNTER_FILE.read_text(encoding="utf-8")) if COUNTER_FILE.exists() else {}
    prob_data = json.loads(PROB_METRICS_FILE.read_text(encoding="utf-8")) if PROB_METRICS_FILE.exists() else {}

    result = ConjectureResult(
        has_counterexample=prob_data.get("has_counterexample", False),
        counterexample_g6=prob_data.get("counterexample_g6", None),
        size_of_counter_example=prob_data.get("size_of_counter_example", None),
        min_order=params.get("min_order"),
        max_order=params.get("max_order"),
        total_mutations=counters.get("total_graphs_muted", 0),
        total_mutations_of_counterexample=prob_data.get("total_mutations_of_counterexample", None),
        score=prob_data.get("score", None),
        total_graphs_generated=counters.get("total_graphs_generated", 0),
        time=elapsed_time,
        total_api_requests=api_requests,
        seed=params.get("seed"),
        subclass=params.get("subclass"),
        cpus=params.get("cpu", 1),
        score_function_path=params.get("script", ""),
        score_function_name=params.get("function", ""),
        approx=params.get("approx", False),
        time_limit=params.get("time_limit", 60.0),
        x_value=prob_data.get("x_value", None),
        y_value=prob_data.get("y_value", None),
    )

    create_log_file(result)

    for temp_file in (COUNTER_FILE, LOCK_FILE, PROB_METRICS_FILE):
        if temp_file.exists():
            temp_file.unlink()

    return result


def create_log_file(result: ConjectureResult) -> None:
    timestamp = datetime.now().strftime("%y%m%d%H%M%S")
    output_dir = BASE_OUT_DIR / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file_path = output_dir / "result.txt"
    result_dict = asdict(result)

    with open(log_file_path, "w", encoding="utf-8") as f:
        f.write("=== RESULTS ===\n")
        f.write("===========================\n\n")
        for key, value in result_dict.items():
            formatted_key = key.replace("_", " ").capitalize()
            formatted_value = value if value is not None else "N/A"
            f.write(f"{formatted_key}: {formatted_value}\n")

    print(f"[SAVE] Results saved to file: {log_file_path}")