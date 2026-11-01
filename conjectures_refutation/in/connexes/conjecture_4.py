def conjecture_4(G):
    import networkx as nx
    from conjectures_refutation.helpers import invariants as inv

    if not inv.is_connected(G):
        return 10.0 + inv.number_of_components(G)

    k_prime = inv.edge_connectivity(G)
    delta = inv.minimum_degree(G)

    rhs = (11 / 4) - (91 / 40) * delta + (11 / 20) * (delta ** 2) - (1 / 40) * (delta ** 3)

    base_score = k_prime - rhs

    epsilon = 1e-4
    heuristic = - (inv.degree_variance(G) * epsilon)

    final_score = base_score + heuristic

    return final_score