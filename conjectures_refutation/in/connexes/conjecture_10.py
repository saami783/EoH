def conjecture_10(G):
    import networkx as nx
    from conjectures_refutation.helpers import invariants as inv

    if not inv.is_connected(G):
        return 10.0 + inv.number_of_components(G)

    delta = inv.minimum_degree(G)
    k_prime = inv.edge_connectivity(G)

    rhs = (75 / 28) + (128 / 85) * k_prime - (11 / 56) * (k_prime ** 2) + (1 / 84) * (k_prime ** 3)

    base_score = rhs - delta

    epsilon = 1e-4
    heuristic = - (inv.degree_variance(G) * epsilon)

    final_score = base_score + heuristic

    return final_score