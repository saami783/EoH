def conjecture_13(G):
    import networkx as nx
    from conjectures_refutation.helpers import invariants as inv

    if not inv.is_connected(G):
        return 10.0 + inv.number_of_components(G)

    omega = len(max(nx.find_cliques(G), key=len))
    k_prime = inv.edge_connectivity(G)

    rhs = (77 / 8) - (20 / 29) * k_prime + (1 / 16) * (k_prime ** 2)

    base_score = rhs - omega

    epsilon = 1e-4
    heuristic = - (inv.average_clustering(G) * epsilon)

    final_score = base_score + heuristic

    return final_score