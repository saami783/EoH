# Placeholder heuristic — replace this with the best heuristic produced by EoH.
# Copy the evolved priority_score function from the EoH results here, then run
# runEval.py to benchmark it and plot the best graph / convergence curve.
import numpy as np
def priority_score(
    conjecture_score: float,
    n_nodes: int,
    n_edges: int,
    max_degree: int,
    min_degree: int,
    diameter: int,
) -> float:
    """Baseline priority heuristic."""
    score = conjecture_score - 0.1 * (n_edges / max(1, n_nodes))
    return float(score)