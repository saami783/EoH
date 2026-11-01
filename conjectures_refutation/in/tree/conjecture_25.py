def conjecture_25(G):
    import networkx as nx
    from conjectures_refutation.helpers import invariants as inv

    if not inv.is_tree(G):
        return 10.0 + inv.number_of_components(G) + abs(inv.size(G) - inv.order(G) + 1)

    rad = inv.radius(G)
    mu = inv.matching_number(G)

    rhs = (4 / 5) + (1 / 5) * mu

    base_score = rad - rhs

    epsilon = 1e-4
    heuristic = - (inv.number_of_leaves(G) * epsilon)

    final_score = base_score + heuristic

    return final_score