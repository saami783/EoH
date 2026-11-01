import sys
import os
import importlib
import importlib.util
from typing import Any
import networkx as nx
from eoh import BaseProblem

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'eoh', 'src'))

import helpers.mutations as mutations
from helpers.database import check_if_counterexample_exists, insert_counterexample
from helpers.results import build_result
import helpers.invariants as invariants_method
from helpers.graph_generation import build_beam
from helpers.increment_counters import increment_counter_mutations

class ConjectureRefutation(BaseProblem):
    """
    EoH problem: Evolve a priority scoring heuristic to guide a beam search
    in order to construct a counterexample to a graph theory conjecture.
    """

    template_program = '''
import math
import numpy as np

def priority_score(
    conjecture_score: float,
    order: int,
    size: int,
    max_degree: int,
    min_degree: int,
    avg_degree: float,
    diameter: int,
    density: float,
    radius: int,
    connectivity: int,
    is_connected: bool,
    is_complete: bool,
    is_tree: bool,
    is_path: bool,
    is_star: bool,
    is_planar: bool,
    is_chordal: bool,
    is_bipartite: bool,
    is_triangle_free: bool,
    is_eulerian: bool,
    is_hamiltonian: bool,
    is_regular: bool,
    is_claw_free: bool,
    is_bull_free: bool,
    is_paw_free: bool,
    is_diamond_free: bool,
) -> float:
    """Score a candidate graph to guide beam search toward a counterexample.

    Called once for each mutated candidate graph at every step of the beam search.
    The harness keeps the top beam_width graphs with the LOWEST finite priority scores.

    Args:
        conjecture_score: float — f(G) = x(G) - y(G) (negative = counterexample found,
                          smaller positive values = closer to violating y(G) <= x(G))
        order:            int — number of vertices |V| (>= 1)
        size:             int — number of edges |E| (>= 0)
        max_degree:       int — maximum vertex degree (>= 0)
        min_degree:       int — minimum vertex degree (>= 0)
        avg_degree:       float — average vertex degree (>= 0.0)
        diameter:         int — graph diameter (can be <= 0 if disconnected)
        density:          float — edge density in [0.0, 1.0]
        radius:           int — graph radius (can be <= 0 if disconnected)
        connectivity:     int — vertex connectivity (>= 0)
        is_*:             bool — structural graph properties (True or False)
    Returns:
        score: float — priority score for the candidate graph (lower = more promising).
    """
    # Default: greedy conjecture score with a small density tie-breaker
    score = conjecture_score - 0.1 * density
    return float(score)
'''

    task_description = (
        "Design a novel priority-scoring function for a beam search algorithm "
        "aimed at refuting a graph theory conjecture of the form y(G) <= x(G). "
        "At each step, candidate graphs are generated via local graph mutations and "
        "evaluated using the conjecture metric f(G) = x(G) - y(G), where f(G) < 0 "
        "indicates a valid counterexample. "
        "The scoring function receives f(G) ('conjecture_score') along with topological "
        "invariants (order, size, degree statistics, diameter, radius, density, connectivity) "
        "and boolean structural properties, and must return a scalar priority score "
        "(lower score = higher priority to be kept in the beam). "
        "The classic baseline greedily ranks graphs by their conjecture_score alone, "
        "which often traps the beam in local optima. "
        "Alternatives include: degree-asymmetry bonuses (e.g., contrasting max_degree and "
        "min_degree), distance-to-order ratios (exploiting diameter and radius), "
        "sparsity or density regularisation, non-linear transformations of conjecture_score, "
        "or conditional penalties/bonuses based on structural boolean invariants. "
        "Avoid utilizing random components so the ranking remains self-consistent. "
        "The goal is to minimise the best conjecture_score f(G) reached during the beam search "
        "and drive it below zero."
    )

    def __init__(self, min_order:int, max_order:int, beam_width:int, max_steps:int, timeout:int,
                 cpu:int, conjecture_score_function_path:str, score_function_name: str, used_counterexamples_db: bool,
                 subclass: str|None):
        super().__init__(timeout=timeout, n_processes=cpu)
        self.min_order = min_order
        self.max_order = max_order
        self.beam_width = beam_width
        self.max_steps = max_steps
        self.score_function_path = conjecture_score_function_path
        self.score_function_name = score_function_name
        self.used_counterexamples_db = used_counterexamples_db
        self.subclass = subclass
        self._loaded_conjecture_fn = None
        self._load_conjecture_function()


    def _load_conjecture_function(self):
        """Dynamically load the user's score function."""

        actual_path = os.path.abspath(self.score_function_path)
        module_dir = os.path.dirname(actual_path)
        module_name = os.path.basename(actual_path)

        if module_name.endswith('.py'):
            module_name = module_name[:-3]

        if module_dir not in sys.path:
            sys.path.insert(0, module_dir)

        custom_module = importlib.import_module(module_name)

        return getattr(custom_module, self.score_function_name)


    def _compute_invariants(self, G: nx.Graph) -> dict[str, int | Any]:
        return {
            "order": invariants_method.order(G),
            "size": invariants_method.size(G),
            "max_degree": invariants_method.maximum_degree(G),
            "min_degree": invariants_method.minimum_degree(G),
            "avg_degree": invariants_method.average_degree(G),
            "diameter": invariants_method.diameter(G),
            "density": invariants_method.density(G),
            "radius": invariants_method.radius(G),
            "connectivity": invariants_method.connectivity(G),
            "is_connected": invariants_method.is_connected(G),
            "is_complete": invariants_method.is_complete(G),
            "is_tree": invariants_method.is_tree(G),
            "is_path": invariants_method.is_path(G),
            "is_star": invariants_method.is_star(G),
            "is_planar": invariants_method.is_planar(G),
            "is_chordal": invariants_method.is_chordal(G),
            "is_bipartite": invariants_method.is_bipartite(G),
            "is_triangle_free": invariants_method.is_triangle_free(G),
            "is_eulerian": invariants_method.is_eulerian(G),
            "is_hamiltonian": invariants_method.is_hamiltonian(G),
            "is_regular": invariants_method.is_regular(G),
            "is_claw_free": invariants_method.is_claw_free(G),
            "is_bull_free": invariants_method.is_bull_free(G),
            "is_paw_free": invariants_method.is_paw_free(G),
            "is_diamond_free": invariants_method.is_diamond_free(G)
        }


    def calculate_conjecture_score(self, G: nx.Graph) -> float:
        if self._loaded_conjecture_fn is None:
            self._loaded_conjecture_fn = self._load_conjecture_function()
        return float(self._loaded_conjecture_fn(G))


    def generate_mutations(self, G: nx.Graph) -> list[nx.Graph]:
        """Generates the x possible mutations for a given graph."""
        mutants = []

        for mutation_func in mutations.MUTATION_REGISTRY.values():
            try:
                mutant = mutation_func(G.copy())
                if mutant is not None and mutant.number_of_nodes() > 0:
                    mutants.append(mutant)
                    increment_counter_mutations()
            except Exception:
                continue

        return mutants


    def evaluate_program(self, program_str: str, callable_func) -> float | None:

        beam = build_beam(self.min_order, self.max_order, self.subclass)
        best_score_found = float('inf')

        try:
            for step in range(self.max_steps):
                candidates = []

                for G in beam:
                    mutants = self.generate_mutations(G)
                    candidates.extend(mutants)

                scored_candidates = []
                for candidate in candidates:
                    conj_score = self.calculate_conjecture_score(candidate)

                    if conj_score < 0:
                        g6_str = nx.to_graph6_bytes(candidate, header=False).decode('ascii').strip()
                        if self.used_counterexamples_db:
                            if check_if_counterexample_exists(g6_str):
                                print(f"Counterexample found but discarded as it is a duplicate. Graph g6: {g6_str}")
                                conj_score = 1e-5
                            else:
                                print(f'Unique counterexample found! Inserting into the database...')
                                print(f'Counterexample graph6 string: {g6_str}')
                                insert_counterexample(g6_str)
                                best_score_found = conj_score
                                print('Generating results report.')
                                build_result(
                                    counterexample_g6=g6_str,
                                    size_of_counter_example=candidate.number_of_nodes(),
                                    total_mutations_of_counterexample=step + 1,
                                    score=best_score_found,
                                )
                                return best_score_found
                        else:
                            best_score_found = conj_score
                            print("Counterexample found!")
                            print('Generating results report.')
                            build_result(
                                counterexample_g6=g6_str,
                                size_of_counter_example=candidate.number_of_nodes(),
                                total_mutations_of_counterexample=step + 1,
                                score=best_score_found,
                            )
                            return best_score_found

                    if conj_score < best_score_found:
                        best_score_found = conj_score

                    invariants = self._compute_invariants(candidate)

                    try:
                        priority = float(callable_func(
                            conjecture_score=conj_score,
                            order=invariants["order"],
                            size=invariants["size"],
                            max_degree=invariants["max_degree"],
                            min_degree=invariants["min_degree"],
                            diameter=invariants["diameter"],
                            avg_degree=invariants["avg_degree"],
                            density=invariants["density"],
                            radius=invariants["radius"],
                            connectivity=invariants["connectivity"],
                            is_connected=invariants["is_connected"],
                            is_complete=invariants["is_complete"],
                            is_tree=invariants["is_tree"],
                            is_path=invariants["is_path"],
                            is_star=invariants["is_star"],
                            is_planar=invariants["is_planar"],
                            is_chordal=invariants["is_chordal"],
                            is_bipartite=invariants["is_bipartite"],
                            is_triangle_free=invariants["is_triangle_free"],
                            is_eulerian=invariants["is_eulerian"],
                            is_hamiltonian=invariants["is_hamiltonian"],
                            is_regular=invariants["is_regular"],
                            is_claw_free=invariants["is_claw_free"],
                            is_bull_free=invariants["is_bull_free"],
                            is_paw_free=invariants["is_paw_free"],
                            is_diamond_free=invariants["is_diamond_free"],
                        ))
                    except Exception:
                        return None

                    scored_candidates.append((priority, candidate))

                scored_candidates.sort(key=lambda x: x[0], reverse=False)
                beam = [cand for priority, cand in scored_candidates[:self.beam_width]]

        except Exception as e:
            return None

        return best_score_found
