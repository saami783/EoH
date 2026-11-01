def conjecture_18(G):
    import networkx as nx
    from conjectures_refutation.helpers import invariants as inv

    if not inv.is_tree(G):
        return 10.0 + inv.number_of_components(G) + abs(inv.size(G) - inv.order(G) + 1)

    tau = len(nx.algorithms.approximation.min_weighted_vertex_cover(G))
    rad = inv.radius(G)

    rhs = -10 + (44 / 3) * rad - 4 * (rad ** 2) + (1 / 3) * (rad ** 3)

    base_score = rhs - tau

    epsilon = 1e-4
    heuristic = - (inv.number_of_leaves(G) * epsilon)

    final_score = base_score + heuristic

    return final_score