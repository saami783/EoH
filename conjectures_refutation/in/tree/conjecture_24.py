def conjecture_24(G):
    import networkx as nx
    from conjectures_refutation.helpers import invariants as inv

    if not inv.is_tree(G):
        return 10.0 + inv.number_of_components(G) + abs(inv.size(G) - inv.order(G) + 1)

    diam = inv.diameter(G)
    mu = inv.matching_number(G)

    rhs = (8 / 5) + (2 / 5) * mu

    base_score = diam - rhs

    epsilon = 1e-4
    heuristic = - (inv.number_of_leaves(G) * epsilon)

    final_score = base_score + heuristic

    return final_score