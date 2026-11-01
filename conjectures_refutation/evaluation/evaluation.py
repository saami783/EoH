import os
import sys
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from prob import ConjectureRefutation


class Evaluation:
    """Thorough evaluation of a Graph Conjecture Refutation heuristic.

    Reuses the ConjectureRefutation problem class from prob.py so that mutations
    and conjecture scoring are defined in a single place.
    """

    def __init__(
        self,
        min_order: int = 5,
        max_order: int = 20,
        beam_width: int = 30,
        max_steps: int = 25,
        conjecture_path: str | None = None,
        score_function=None,
        is_inf: bool = True,
    ):
        self.problem = ConjectureRefutation(
            min_order=min_order,
            max_order=max_order,
            beam_width=beam_width,
            max_steps=max_steps,
            conjecture_path=conjecture_path,
            score_function=score_function,
            is_inf=is_inf,
        )
        self.min_order = min_order
        self.max_order = max_order
        self.beam_width = beam_width
        self.max_steps = max_steps
        self.is_inf = is_inf

    def evaluate(self, heuristic_fn) -> dict:
        """Run the beam search guided by heuristic_fn and return detailed stats.

        Args:
            heuristic_fn: callable(conjecture_score, n_nodes, n_edges,
                                   max_degree, min_degree, diameter) -> float

        Returns:
            dict with keys:
                step_best_scores  – list of best conjecture_score at each step
                step_mean_scores  – list of mean conjecture_score in the beam at each step
                best_score        – best overall conjecture_score found
                best_graph        – networkx.Graph achieving best_score
                counterexample    – bool, True if a counterexample was found
                step_found        – step index where best_score was first reached
        """
        beam = [
            nx.path_graph(n)
            for n in range(self.min_order, self.max_order)
        ]

        best_score = float('inf') if self.is_inf else -float('inf')
        best_graph = beam[0] if beam else nx.Graph()
        step_found = 0

        step_best_scores = []
        step_mean_scores = []

        for step in range(1, self.max_steps + 1):
            candidates = []
            for G in beam:
                mutants = self.problem.generate_mutations(G)
                candidates.extend(mutants)

            # Fallback if generate_mutations is not yet implemented and returns []
            if not candidates:
                candidates = list(beam)

            scored_candidates = []
            current_step_scores = []

            for candidate in candidates:
                conj_score = self.problem.calculate_conjecture_score(candidate)
                current_step_scores.append(conj_score)

                is_better = (conj_score < best_score) if self.is_inf else (conj_score > best_score)
                if is_better:
                    best_score = conj_score
                    best_graph = candidate.copy()
                    step_found = step

                n_nodes = candidate.number_of_nodes()
                n_edges = candidate.number_of_edges()
                degrees = [d for _, d in candidate.degree()]
                max_d = max(degrees) if degrees else 0
                min_d = min(degrees) if degrees else 0
                diam = nx.diameter(candidate) if (n_nodes > 0 and nx.is_connected(candidate)) else 0

                priority = float(heuristic_fn(
                    conjecture_score=conj_score,
                    n_nodes=n_nodes,
                    n_edges=n_edges,
                    max_degree=max_d,
                    min_degree=min_d,
                    diameter=diam,
                ))
                scored_candidates.append((priority, candidate))

            scored_candidates.sort(key=lambda x: x[0], reverse=False)
            beam = [cand for _, cand in scored_candidates[:self.beam_width]]

            if current_step_scores:
                step_best = min(current_step_scores) if self.is_inf else max(current_step_scores)
                step_mean = float(np.mean(current_step_scores))
            else:
                step_best = best_score
                step_mean = best_score

            step_best_scores.append(float(step_best))
            step_mean_scores.append(float(step_mean))

        is_ce = (best_score < 0.0) if self.is_inf else (best_score > 0.0)

        return {
            'step_best_scores': step_best_scores,
            'step_mean_scores': step_mean_scores,
            'best_score':       float(best_score),
            'best_graph':       best_graph,
            'counterexample':   bool(is_ce),
            'step_found':       int(step_found),
        }

    def plot_scores(
        self,
        stats: dict,
        label: str = 'heuristic',
        save_path: str = 'scores.png',
    ) -> None:
        """Plot convergence of the conjecture score across mutation steps."""
        best_scores = np.array(stats['step_best_scores'])
        mean_scores = np.array(stats['step_mean_scores'])
        steps = np.arange(1, len(best_scores) + 1)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(steps, best_scores, marker='o', color='#2ecc71', linewidth=2.0,
                label='Best step score $f(G)$', zorder=3)
        ax.plot(steps, mean_scores, linestyle='--', color='#3498db', linewidth=1.5,
                label='Mean candidate score $f(G)$', zorder=2)
        ax.axhline(0.0, color='#e74c3c', linewidth=1.5, linestyle=':',
                   label='Counterexample threshold ($f(G) = 0$)', zorder=2)

        ax.set_xlabel('Mutation Step', fontsize=12)
        ax.set_ylabel('Conjecture Score $f(G) = x(G) - y(G)$', fontsize=12)
        ax.set_title(f'Beam Search Convergence — {label}', fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.5, zorder=0)

        status = 'COUNTEREXAMPLE FOUND' if stats['counterexample'] else 'No Counterexample'
        ax.text(
            0.98, 0.96,
            f"{status}\nBest: {stats['best_score']:+.4f} (step {stats['step_found']})",
            transform=ax.transAxes, ha='right', va='top', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='#bdc3c7', alpha=0.9),
        )

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.savefig(save_path.rsplit('.', 1)[0] + '.pdf')
        plt.close(fig)
        print(f'  → convergence plot : {save_path}')

    def render_best_graph(
        self,
        stats: dict,
        label: str = 'heuristic',
        save_path: str = 'best_graph.png',
    ) -> None:
        """Render and save the best graph (or counterexample) found."""
        G = stats['best_graph']
        fig, ax = plt.subplots(figsize=(7, 6))

        if G.number_of_nodes() > 0:
            pos = nx.kamada_kawai_layout(G) if G.number_of_nodes() <= 50 else nx.spring_layout(G, seed=42)
            node_color = '#e74c3c' if stats['counterexample'] else '#3498db'
            nx.draw_networkx(
                G, pos=pos, ax=ax,
                node_color=node_color,
                node_size=350,
                font_color='white',
                font_size=9,
                edge_color='#7f8c8d',
                width=1.5,
            )

        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        ax.set_title(
            f"Best Graph — {label}\n"
            f"$|V|={n_nodes}, |E|={n_edges}$, Score $f(G) = {stats['best_score']:+.4f}$",
            fontsize=12,
        )
        ax.axis('off')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.savefig(save_path.rsplit('.', 1)[0] + '.pdf')
        plt.close(fig)
        print(f'  → best graph plot  : {save_path}')