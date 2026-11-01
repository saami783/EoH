def conjecture_6(G):
    import networkx as nx
    from conjectures_refutation.helpers import invariants as inv

    if not inv.is_connected(G):
        return 10.0 + inv.number_of_components(G)

    t = inv.triangle_number(G)
    tau = len(nx.algorithms.approximation.min_weighted_vertex_cover(G))

    rhs = -(572 / 7) + (2039 / 21) * tau - (114 / 7) * (tau ** 2) + (19 / 21) * (tau ** 3)

    base_score = rhs - t

    epsilon = 1e-4
    heuristic = - (inv.average_clustering(G) * epsilon)

    final_score = base_score + heuristic

    return final_score