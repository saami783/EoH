from typing import Callable, Dict
import networkx as nx

def _get_next_node_id(G: nx.Graph, count: int = 1) -> list:
    """Retourne une liste d'identifiants uniques pour de nouveaux nœuds."""
    int_nodes = [n for n in G.nodes if isinstance(n, int)]
    base = max(int_nodes, default=-1) + 1
    return [base + i for i in range(count)]


def mutation_add_edge(G: nx.Graph) -> nx.Graph:
    """Ajoute la première arête manquante dans l'ordre lexicographique des paires."""
    H = G.copy()
    nodes = sorted(H.nodes())
    for i, u in enumerate(nodes):
        for v in nodes[i + 1 :]:
            if not H.has_edge(u, v):
                H.add_edge(u, v)
                return H
    return H


def mutation_remove_edge(G: nx.Graph) -> nx.Graph:
    """Supprime la première arête triée canoniquement."""
    H = G.copy()
    edges = sorted(tuple(sorted((u, v))) for u, v in H.edges())
    if edges:
        u, v = edges[0]
        H.remove_edge(u, v)
    return H


def mutation_add_vertex(G: nx.Graph) -> nx.Graph:
    """Ajoute un sommet et le relie au premier nœud existant."""
    H = G.copy()
    new_node = _get_next_node_id(H, 1)[0]
    H.add_node(new_node)
    nodes = sorted(G.nodes())
    if nodes:
        H.add_edge(new_node, nodes[0])
    return H


def mutation_remove_vertex(G: nx.Graph) -> nx.Graph:
    """Supprime le premier sommet dans l'ordre de tri."""
    H = G.copy()
    nodes = sorted(H.nodes())
    if nodes:
        H.remove_node(nodes[0])
    return H


def mutation_subdivision(G: nx.Graph) -> nx.Graph:
    """Subdivise la première arête en insérant un nouveau sommet au milieu."""
    H = G.copy()
    edges = sorted(tuple(sorted((u, v))) for u, v in H.edges())
    if edges:
        u, v = edges[0]
        new_node = _get_next_node_id(H, 1)[0]
        H.remove_edge(u, v)
        H.add_edge(u, new_node)
        H.add_edge(new_node, v)
    return H


def mutation_contraction(G: nx.Graph) -> nx.Graph:
    """Contracte la première arête (u, v) en fusionnant v dans u."""
    H = G.copy()
    edges = sorted(tuple(sorted((u, v))) for u, v in H.edges())
    if edges:
        u, v = edges[0]
        H = nx.contracted_nodes(H, u, v, self_loops=False)
    return H


def mutation_replace_vertex_by_path(G: nx.Graph) -> nx.Graph:
    """Remplace le premier sommet par un chemin P_k (où k = degré du sommet)."""
    H = G.copy()
    nodes = sorted(H.nodes())
    if not nodes:
        return H

    v = nodes[0]
    neighbors = sorted(H.neighbors(v))
    k = len(neighbors)

    if k == 0:
        return H

    new_nodes = _get_next_node_id(H, k)
    H.remove_node(v)

    for i in range(k - 1):
        H.add_edge(new_nodes[i], new_nodes[i + 1])

    for i, neighbor in enumerate(neighbors):
        H.add_edge(new_nodes[i], neighbor)

    return H


def mutation_replace_vertex_by_star(G: nx.Graph) -> nx.Graph:
    """Remplace le premier sommet par une étoile S_k connectée à ses voisins."""
    H = G.copy()
    nodes = sorted(H.nodes())
    if not nodes:
        return H

    v = nodes[0]
    neighbors = sorted(H.neighbors(v))
    k = len(neighbors)

    if k == 0:
        return H

    new_nodes = _get_next_node_id(H, k + 1)
    center = new_nodes[0]
    leaves = new_nodes[1:]

    H.remove_node(v)

    for leaf in leaves:
        H.add_edge(center, leaf)

    for leaf, neighbor in zip(leaves, neighbors):
        H.add_edge(leaf, neighbor)

    return H


def mutation_replace_vertex_by_clique(G: nx.Graph) -> nx.Graph:
    """Remplace le premier sommet par une clique K_k connectée à ses voisins."""
    H = G.copy()
    nodes = sorted(H.nodes())
    if not nodes:
        return H

    v = nodes[0]
    neighbors = sorted(H.neighbors(v))
    k = len(neighbors)

    if k == 0:
        return H

    new_nodes = _get_next_node_id(H, k)
    H.remove_node(v)

    for i in range(k):
        for j in range(i + 1, k):
            H.add_edge(new_nodes[i], new_nodes[j])

    for new_node, neighbor in zip(new_nodes, neighbors):
        H.add_edge(new_node, neighbor)

    return H


def mutation_replace_vertex_by_polyhedral(G: nx.Graph) -> nx.Graph:
    """Remplace le premier sommet par un tétraèdre (K4) et distribue les voisins de façon cyclique."""
    H = G.copy()
    nodes = sorted(H.nodes())
    if not nodes:
        return H

    v = nodes[0]
    neighbors = sorted(H.neighbors(v))
    new_nodes = _get_next_node_id(H, 4)

    H.remove_node(v)

    for i in range(4):
        for j in range(i + 1, 4):
            H.add_edge(new_nodes[i], new_nodes[j])

    for i, neighbor in enumerate(neighbors):
        target_node = new_nodes[i % 4]
        H.add_edge(target_node, neighbor)

    return H


def mutation_bipartition_neighborhood(G: nx.Graph) -> nx.Graph:
    """Scinde le premier sommet en deux (v1, v2) et répartit ses voisins selon la parité de leur index."""
    H = G.copy()
    nodes = sorted(H.nodes())
    if not nodes:
        return H

    v = nodes[0]
    neighbors = sorted(H.neighbors(v))
    v1, v2 = _get_next_node_id(H, 2)

    H.remove_node(v)
    H.add_edge(v1, v2)

    for i, neighbor in enumerate(neighbors):
        if i % 2 == 0:
            H.add_edge(v1, neighbor)
        else:
            H.add_edge(v2, neighbor)

    return H

MutationFunction = Callable[[nx.Graph], nx.Graph]

MUTATION_REGISTRY: Dict[str, MutationFunction] = {
    "add_edge": mutation_add_edge,
    "remove_edge": mutation_remove_edge,
    "add_vertex": mutation_add_vertex,
    "remove_vertex": mutation_remove_vertex,
    "subdivision": mutation_subdivision,
    "contraction": mutation_contraction,
    "replace_vertex_by_path": mutation_replace_vertex_by_path,
    "replace_vertex_by_star": mutation_replace_vertex_by_star,
    "replace_vertex_by_clique": mutation_replace_vertex_by_clique,
    "replace_vertex_by_polyhedral": mutation_replace_vertex_by_polyhedral,
    "bipartition_neighborhood": mutation_bipartition_neighborhood,
}