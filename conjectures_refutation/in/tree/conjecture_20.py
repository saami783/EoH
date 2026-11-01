def conjecture_20(G):
    import networkx as nx
    from conjectures_refutation.helpers import invariants as inv

    if not inv.is_tree(G):
        return 10.0 + inv.number_of_components(G) + abs(inv.size(G) - inv.order(G) + 1)

    diam = inv.diameter(G)
    gamma = len(nx.algorithms.approximation.min_weighted_dominating_set(G))

    rhs = (5 / 8) + (2 / 5) * gamma

    base_score = diam - rhs

    epsilon = 1e-4
    heuristic = - (inv.number_of_leaves(G) * epsilon)

    final_score = base_score + heuristic

    return final_score