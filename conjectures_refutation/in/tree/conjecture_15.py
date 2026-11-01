def conjecture_15(G):
    import networkx as nx
    from conjectures_refutation.helpers import invariants as inv

    if not inv.is_tree(G):
        return 10.0 + inv.number_of_components(G) + abs(inv.size(G) - inv.order(G) + 1)

    gamma = len(nx.algorithms.approximation.min_weighted_dominating_set(G))
    diam = inv.diameter(G)

    rhs = -(100 / 9) + (149 / 18) * diam - (11 / 9) * (diam ** 2) + (1 / 18) * (diam ** 3)

    base_score = rhs - gamma

    epsilon = 1e-4
    heuristic = - (inv.number_of_leaves(G) * epsilon)

    final_score = base_score + heuristic

    return final_score