def conjecture_2(G):
    import networkx as nx
    from conjectures_refutation.helpers import invariants as inv

    if not inv.is_connected(G):
        return 10.0 + inv.number_of_components(G)

    avg = inv.average_degree(G)
    delta = inv.minimum_degree(G)

    rhs = (67 / 9) - (7 / 60) * delta + (7 / 97) * (delta ** 2)
    base_score = rhs - avg

    epsilon = 1e-4

    variance_bonus = - (inv.degree_variance(G) * epsilon)

    delta_penalty = abs(delta - 1) * epsilon * 10

    final_score = base_score + variance_bonus + delta_penalty

    return final_score