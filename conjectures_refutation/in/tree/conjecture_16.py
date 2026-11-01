def conjecture_16(G):
    import networkx as nx
    from conjectures_refutation.helpers import invariants as inv

    if not inv.is_tree(G):
        return 10.0 + inv.number_of_components(G) + abs(inv.size(G) - inv.order(G) + 1)

    tau = len(nx.algorithms.approximation.min_weighted_vertex_cover(G))
    diam = inv.diameter(G)

    rhs = -(47 / 5) + (149 / 20) * diam - (11 / 10) * (diam ** 2) + (1 / 20) * (diam ** 3)

    base_score = rhs - tau

    epsilon = 1e-4
    heuristic = - (inv.number_of_leaves(G) * epsilon)

    final_score = base_score + heuristic

    return final_score