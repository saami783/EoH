def conjecture_1(G):
    import networkx as nx
    from helpers import invariants as inv

    if not inv.is_connected(G):
        return 10.0 + inv.number_of_components(G)

    d = inv.density(G)
    rad = inv.radius(G)

    rhs = (65 / 98) + (18 / 29) * rad - (17 / 53) * (rad ** 2) + (1 / 28) * (rad ** 3)
    base_score = rhs - d

    epsilon = 1e-4
    heuristic = - (inv.degree_variance(G) * epsilon)

    distance_to_optimal_radius = abs(rad - 4.5)
    radius_penalty = distance_to_optimal_radius * epsilon * 10

    final_score = base_score + heuristic + radius_penalty

    return final_score