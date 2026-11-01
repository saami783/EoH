def conjecture_7(G):
    import networkx as nx
    from conjectures_refutation.helpers import invariants as inv

    if not inv.is_connected(G):
        return 10.0 + inv.number_of_components(G)

    omega = len(max(nx.find_cliques(G), key=len))
    k = inv.vertex_connectivity(G)

    rhs = (262 / 27) - (43 / 54) * k + (5 / 54) * (k ** 2)

    base_score = rhs - omega

    epsilon = 1e-4
    heuristic = - (inv.average_clustering(G) * epsilon)

    final_score = base_score + heuristic

    return final_score