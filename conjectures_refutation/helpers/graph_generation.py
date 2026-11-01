from typing import List
import networkx as nx
from .subclass import subclass_function
from .increment_counters import increment_counter_graph_generation

def _build_graph(subclass: str, n: int) -> nx.Graph:
  if subclass not in subclass_function:
    raise ValueError(f"Subclass '{subclass}' does not exists. Please implements it in conjectures_refutation/helpers/subclass.py")

  func = subclass_function[subclass]

  try:
    return func(n)
  except Exception:
    return None


def build_beam(min_order: int, max_order: int, subclass: str|None = None) -> List[nx.Graph]:
    beam = []
    if subclass is not None:
        for n in range(min_order, max_order+1):
            G = _build_graph(subclass, n)
            if G is not None:
                beam.append(G)
                increment_counter_graph_generation()
    else:
        for sub_name in subclass_function:
            for n in range(min_order, max_order+1):
                G = _build_graph(sub_name, n)
                if G is not None:
                    beam.append(G)
                    increment_counter_graph_generation()
    return beam

