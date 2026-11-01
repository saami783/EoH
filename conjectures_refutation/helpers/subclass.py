import networkx as nx

def build_tree_graph(order: int) -> nx.Graph:
    return nx.random_labeled_tree(order)

def build_connected_graph(order: int) -> nx.Graph:
    print(f"[DEBUG] génération d'un graphe connecté à {order} sommets...")
    if order <= 3:
        raise ValueError(f"L'ordre du graphe doit être supérieur ou égal à 3 pour la génération d'un graphe connexe.")
    return nx.cycle_graph(order)

def build_planar_graph(order: int):
    print(f"[DEBUG] génération d'un graphe planaire à {order} sommets...")
    if order < 1:
        raise ValueError(f"L'ordre du graphe doit être supérieur à 1 pour la génération d'un graphe planaire.")
    return nx.wheel_graph(order)

def build_bipartite_graph(order: int):
  print(f"[DEBUG] génération d'un graphe biparti à {order} sommets...")
  if order < 2:
    raise ValueError(
        f"L'ordre du graphe doit être supérieur à 2 pour la génération d'un"
        f" graphe bipartie."
    )
  n1 = order // 2
  n2 = order - n1
  return nx.complete_bipartite_graph(n1, n2)

def build_claw_free_graph(order: int):
    print(f"[DEBUG] génération d'un graphe sans griffe à {order} sommets...")
    if order < 1:
        raise ValueError(f"L'ordre du graphe doit être supérieur à 1 pour la génération d'un graphe sans griffe.")
    return nx.complete_graph(order)

subclass_function = {
    "connected": build_connected_graph,
    "tree": build_tree_graph,
    "planar": build_planar_graph,
    "bipartite": build_bipartite_graph,
    "claw_free": build_claw_free_graph,
    # "forest": build_forest_graph,
    # "star": build_star_graph,
    # "diamond_free": build_diamond_free_graph,
    # "regular": build_regular_graph,
}
