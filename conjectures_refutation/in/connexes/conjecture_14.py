def conjecture_14(G):
    import networkx as nx
    from conjectures_refutation.helpers import invariants as inv

    if not inv.is_connected(G):
        return 10.0 + inv.number_of_components(G)

    k = inv.vertex_connectivity(G)
    k_prime = inv.edge_connectivity(G)

    base_score = k - k_prime + 3

    epsilon = 1e-4
    heuristic = - (inv.density(G) * epsilon)

    final_score = base_score + heuristic

    return final_score