# -------
# Evaluation code for EoH on Graph Conjecture Refutation
# -------

import importlib
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

from evaluation import Evaluation

MIN_ORDER = 5
MAX_ORDER = 20
BEAM_WIDTH = 30
MAX_STEPS = 25


def run(
    label: str,
    module_name: str,
    eva: Evaluation,
    results_file,
    visualize: bool = True,
):
    mod = importlib.import_module(module_name)
    mod = importlib.reload(mod)

    t0 = time.time()
    stats = eva.evaluate(mod.priority_score)
    elapsed = time.time() - t0

    G = stats['best_graph']
    ce_str = "YES" if stats['counterexample'] else "NO"

    line = (
        f"{label:30s} | "
        f"best_score: {stats['best_score']:+10.4f}  "
        f"CE: {ce_str:3s}  "
        f"step: {stats['step_found']:3d}  "
        f"|V|: {G.number_of_nodes():3d}  "
        f"|E|: {G.number_of_edges():4d}  "
        f"time: {elapsed:.2f}s"
    )
    print(line)
    results_file.write(line + "\n")

    if visualize:
        slug = label.replace(' ', '_').replace('(', '').replace(')', '').strip('_')
        out_dir = os.path.dirname(os.path.abspath(results_file.name))
        eva.plot_scores(
            stats,
            label=label,
            save_path=os.path.join(out_dir, f'scores_{slug}.png'),
        )
        eva.render_best_graph(
            stats,
            label=label,
            save_path=os.path.join(out_dir, f'best_graph_{slug}.png'),
        )


if __name__ == "__main__":
    eva = Evaluation(
        min_order=MIN_ORDER,
        max_order=MAX_ORDER,
        beam_width=BEAM_WIDTH,
        max_steps=MAX_STEPS,
    )

    results_path = os.path.join(os.path.dirname(__file__), "results.txt")
    with open(results_path, "w") as f:
        header = (
            f"Conjecture Refutation evaluation — orders [{MIN_ORDER}, {MAX_ORDER}), "
            f"beam_width={BEAM_WIDTH}, max_steps={MAX_STEPS}\n"
            + "-" * 95
        )
        print(header)
        f.write(header + "\n")

        # Baseline heuristic (template)
        run("baseline", "heuristic", eva, f, visualize=True)

        # To benchmark your EoH-evolved heuristic, copy its priority_score
        # function into heuristic.py and re-run this script.