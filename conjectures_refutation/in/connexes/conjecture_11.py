def conjecture_11(G):
    import networkx as nx
    from conjectures_refutation.helpers import invariants as inv

    if not inv.is_connected(G):
        return 10.0 + inv.number_of_components(G)

    avg = inv.average_degree(G)
    k_prime = inv.edge_connectivity(G)

    rhs = (67 / 9) - (7 / 60) * k_prime + (7 / 97) * (k_prime ** 2)

    base_score = rhs - avg

    epsilon = 1e-4
    heuristic = - (inv.degree_variance(G) * epsilon)

    final_score = base_score + heuristic

    return final_score