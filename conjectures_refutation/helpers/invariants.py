from collections import deque
from itertools import combinations, permutations
from typing import Dict, Optional
import math
import networkx as nx
import numpy as np
import pulp

_CBC_SEED = 42
_CBC_THREADS = 1

def set_cbc_solver_seed(seed: Optional[int]) -> None:
    """Configure the seed used by all CBC solves (``None`` resets to default)."""

    global _CBC_SEED
    _CBC_SEED = 42 if seed is None else int(seed)


def _cbc_solver() -> pulp.PULP_CBC_CMD:
    return pulp.PULP_CBC_CMD(
        msg=False,
        threads=_CBC_THREADS,
        options=[f"randomSeed={_CBC_SEED}"]
    )
import scipy

def contains_induced_subgraph(G, H):
    """Return ``True`` when ``G`` contains an induced subgraph isomorphic to ``H``."""

    n_H = H.number_of_nodes()
    if G.number_of_nodes() < n_H:
        return False

    graph_matcher = nx.algorithms.isomorphism.GraphMatcher
    for nodes in combinations(G.nodes(), n_H):
        subgraph = G.subgraph(nodes)
        matcher = graph_matcher(subgraph, H)
        if matcher.is_isomorphic():
            return True
    return False


def get_sorted_nodes(G: nx.Graph):
    """Return the graph nodes in a deterministic sorted order."""

    return sorted(G.nodes())


# Property predicates

def is_connected(G):
    """Return ``True`` when graph ``G`` is connected."""

    return nx.is_connected(G)


def is_complete(G):
    """Return ``True`` when ``G`` is complete (every pair of nodes is adjacent)."""

    n = G.number_of_nodes()
    # An undirected complete graph has n(n-1)/2 edges.
    return G.number_of_edges() == n * (n - 1) // 2


def is_tree(G):
    """Return ``True`` when ``G`` is a tree (connected and cycle-free)."""

    return nx.is_tree(G)


def is_path(G):
    """Return ``True`` when ``G`` is a simple path."""

    n = G.number_of_nodes()
    if n == 0:
        return False
    if n == 1:
        return True
    if not is_connected(G):
        return False

    degree_counts = [degree for _, degree in G.degree()]
    if n == 2:
        # Only one edge may exist between the two nodes.
        return G.number_of_edges() == 1

    # For n >= 3 we need two leaves and every internal node of degree two.
    return degree_counts.count(1) == 2 and degree_counts.count(2) == n - 2


def is_star(G):
    """Return ``True`` when ``G`` is a star graph."""

    n = G.number_of_nodes()
    if n == 0:
        return False
    if n == 1:
        return True
    if not is_connected(G):
        return False

    degrees = [degree for _, degree in G.degree()]
    return (n - 1) in degrees and degrees.count(1) == n - 1


def is_planar(G):
    """Return ``True`` when ``G`` is planar."""

    is_planar_flag, _ = nx.check_planarity(G)
    return is_planar_flag


def is_chordal(G):
    """Return ``True`` when ``G`` is chordal."""

    return nx.is_chordal(G)


def is_bipartite(G):
    """Return ``True`` when ``G`` is bipartite."""

    return nx.is_bipartite(G)


def is_triangle_free(G):
    """Return ``True`` when ``G`` is triangle-free."""

    triangles = nx.triangles(G)
    # Each triangle is counted three times across its incident vertices.
    return sum(triangles.values()) == 0


def is_eulerian(G):
    """Return ``True`` when ``G`` admits an Eulerian circuit."""

    return nx.is_eulerian(G)


def is_hamiltonian(G):
    """Brute-force search deciding whether ``G`` admits a Hamiltonian cycle."""

    n = G.number_of_nodes()
    if n == 0:
        return False
    if n == 1:
        return True

    if not is_connected(G):
        return False

    nodes = list(G.nodes())
    start, *rest = nodes

    for perm in permutations(rest):
        cycle = [start, *perm, start]
        if all(G.has_edge(u, v) for u, v in zip(cycle, cycle[1:])):
            return True
    return False


def is_regular(G):
    """Return ``True`` when all vertices in ``G`` share the same degree."""

    if G.number_of_nodes() == 0:
        return True
    degrees = {degree for _, degree in G.degree()}
    return len(degrees) == 1

def is_claw_free(G):
    """Return ``True`` when ``G`` contains no induced claw (``K_{1,3}``)."""

    for u in G.nodes():
        neighbors = list(G.neighbors(u))
        if len(neighbors) < 3:
            continue
        for v, w, x in combinations(neighbors, 3):
            if not (G.has_edge(v, w) or G.has_edge(v, x) or G.has_edge(w, x)):
                return False
    return True


def is_bull_free(G):
    """Return ``True`` when ``G`` contains no induced bull (five vertices)."""

    bull = nx.Graph()
    bull.add_nodes_from([1, 2, 3, 4, 5])
    bull.add_edges_from([(1, 2), (2, 3), (3, 1), (2, 4), (3, 5)])

    return not contains_induced_subgraph(G, bull)


def is_paw_free(G):
    """Return ``True`` when ``G`` contains no induced paw (triangle with a pendant vertex)."""

    paw = nx.Graph()
    paw.add_nodes_from([1, 2, 3, 4])
    paw.add_edges_from([(1, 2), (2, 3), (3, 1), (3, 4)])

    return not contains_induced_subgraph(G, paw)


def is_diamond_free(G):
    """Return ``True`` when ``G`` contains no induced diamond (``K_4`` minus one edge)."""

    diamond = nx.Graph()
    diamond.add_nodes_from([1, 2, 3, 4])
    diamond.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 1), (2, 4)])

    return not contains_induced_subgraph(G, diamond)


#####################################
# Polynomial-time invariants        #
#####################################


def order(G):
    """Return the order of ``G`` (number of vertices)."""

    return G.number_of_nodes()


def size(G):
    """Return the size of ``G`` (number of edges)."""

    return G.number_of_edges()


def diameter(G):
    """Return the diameter of ``G`` or ``float('inf')`` when disconnected."""

    if G.number_of_nodes() == 0:
        return 0
    if nx.is_connected(G):
        return nx.diameter(G)
    return float("inf")


def radius(G):
    """Return the radius of ``G`` or ``float('inf')`` when disconnected."""

    if G.number_of_nodes() == 0:
        return 0
    if nx.is_connected(G):
        return nx.radius(G)
    return float("inf")


def density(G):
    """Return the density of ``G``."""

    return nx.density(G)


def number_of_components(G):
    """Return the number of connected components of ``G``."""

    if G.number_of_nodes() == 0:
        return 0
    return nx.number_connected_components(G)


def largest_component_ratio(G):
    """Return ``|C_max| / |V(G)|`` for the largest connected component."""

    n = G.number_of_nodes()
    if n == 0:
        return 0.0
    return max(len(component) for component in nx.connected_components(G)) / n


def minimum_degree(G):
    """Return the minimum vertex degree in ``G``."""

    if G.number_of_nodes() == 0:
        return 0
    return min(degree for _, degree in G.degree())


def maximum_degree(G):
    """Return the maximum vertex degree in ``G``."""

    if G.number_of_nodes() == 0:
        return 0
    return max(degree for _, degree in G.degree())


def average_degree(G):
    """Return the average degree of ``G``."""

    n = G.number_of_nodes()
    if n == 0:
        return 0
    return sum(degree for _, degree in G.degree()) / n


def degree_variance(G):
    """Return the variance of the degree sequence of ``G``."""

    n = G.number_of_nodes()
    if n == 0:
        return 0.0
    degrees = [degree for _, degree in G.degree()]
    mean = sum(degrees) / n
    return sum((degree - mean) ** 2 for degree in degrees) / n


def average_clustering(G):
    """Return the average clustering coefficient of ``G``."""

    if G.number_of_nodes() <= 1:
        return 0.0
    return nx.average_clustering(G)


def number_of_leaves(G):
    """Return the number of leaves of ``G``."""

    return sum(1 for _, degree in G.degree() if degree == 1)


def number_of_articulation_points(G):
    """Return the number of articulation points of ``G``."""

    if G.number_of_nodes() <= 1:
        return 0
    return sum(1 for _ in nx.articulation_points(G))


def number_of_bridges(G):
    """Return the number of bridges of ``G``."""

    if G.number_of_edges() == 0:
        return 0
    return sum(1 for _ in nx.bridges(G))

def girth(G):
    """Return the length of the shortest simple cycle or ``None`` for acyclic graphs."""

    if G.number_of_edges() == 0:
        return None

    girth_value = float("inf")
    for source in G.nodes():
        distances = {source: 0}
        parents = {source: None}
        queue = deque([source])

        while queue:
            current = queue.popleft()
            for neighbor in G.neighbors(current):
                if neighbor not in distances:
                    distances[neighbor] = distances[current] + 1
                    parents[neighbor] = current
                    queue.append(neighbor)
                elif parents[current] != neighbor:
                    cycle_length = distances[current] + distances[neighbor] + 1
                    girth_value = min(girth_value, cycle_length)
        if girth_value == 3:  # cannot do better than a triangle
            return 3
    return None if girth_value == float("inf") else girth_value


def circumference(G):
    """Return the length (in vertices) of the longest simple cycle in ``G``."""

    if G.number_of_edges() == 0:
        return 0

    DG = G.to_directed()
    cycles = list(nx.simple_cycles(DG))
    # Remove duplicates caused by rotational symmetries.
    normalized_cycles = set()
    for cycle in cycles:
        n = len(cycle)
        rotations = [tuple(cycle[i:] + cycle[:i]) for i in range(n)]
        normalized = min(rotations)
        normalized_cycles.add(normalized)
    if not normalized_cycles:
        return 0
    return max(len(cycle) for cycle in normalized_cycles)


#####################################
# NP-hard invariants (exact or brute force)
#####################################

def treewidth(G):
    """Return the treewidth upper bound given by NetworkX' minimum-degree heuristic."""

    tw, _ = nx.approximation.treewidth_min_degree(G)
    return tw


def longest_path(G):
    """Return the length (number of edges) of the longest simple path in ``G``."""

    longest = 0

    def dfs(node, visited, length):
        nonlocal longest
        longest = max(longest, length)
        for neighbor in G.neighbors(node):
            if neighbor not in visited:
                dfs(neighbor, visited | {neighbor}, length + 1)

    for node in G.nodes():
        dfs(node, {node}, 0)
    return longest


# def longest_induced_path(G):
#     """Return the length (edges) of the longest induced simple path in ``G``."""
#
#     longest = 0
#
#     def is_induced(path):
#         length = len(path)
#         for i in range(length):
#             for j in range(i + 2, length):
#                 if G.has_edge(path[i], path[j]):
#                     return False
#         return True
#
#     def dfs(path, visited):
#         nonlocal longest
#         if not is_induced(path):
#             return
#         longest = max(longest, len(path) - 1)
#         current = path[-1]
#         for neighbor in G.neighbors(current):
#             if neighbor not in visited:
#                 dfs(path + [neighbor], visited | {neighbor})
#
#     for node in G.nodes():
#         dfs([node], {node})
#     return longest

def longest_induced_path(G):
    """
    Retourne le nombre de SOMMETS du plus long chemin induit simple dans G.
    """
    max_vertices = 0
    def dfs(current_path_last, current_path_set, current_length):
        nonlocal max_vertices

        if current_length > max_vertices:
            max_vertices = current_length

        u = current_path_last

        for v in G.neighbors(u):
            if v in current_path_set:
                continue

            neighbors_in_path = sum(1 for n in G.neighbors(v) if n in current_path_set)

            # neighbors_in_path doit être exactement 1 (c'est 'u')
            if neighbors_in_path == 1:
                dfs(v, current_path_set | {v}, current_length + 1)

    nodes = list(G.nodes())
    for node in nodes:
        dfs(node, {node}, 1)

    return max_vertices

def longest_induced_cycle(G):
    """Return the length (vertices) of the longest induced cycle in ``G``."""

    if G.number_of_edges() == 0:
        return 0

    DG = G.to_directed()
    cycles = list(nx.simple_cycles(DG))
    normalized_cycles = set()
    longest = 0
    for cycle in cycles:
        n = len(cycle)
        rotations = [tuple(cycle[i:] + cycle[:i]) for i in range(n)]
        normalized = min(rotations)
        if normalized in normalized_cycles:
            continue
        normalized_cycles.add(normalized)
        # Check for chords by skipping immediate neighbours in the cycle.
        is_induced = True
        cycle_nodes = list(normalized)
        length = len(cycle_nodes)
        for i in range(length):
            for j in range(i + 2, i + length - 1):
                if G.has_edge(cycle_nodes[i], cycle_nodes[j % length]):
                    is_induced = False
                    break
            if not is_induced:
                break
        if is_induced:
            longest = max(longest, length)
    return longest


def chromatic_index(G):
    """Return the edge chromatic number of ``G`` using a MILP fallback."""

    if G.number_of_edges() == 0:
        return 0

    degrees = {node: deg for node, deg in G.degree()}
    delta = max(degrees.values())

    if nx.is_bipartite(G):
        return delta

    n = G.number_of_nodes()
    m = G.number_of_edges()
    if m == n * (n - 1) // 2:
        return n - 1 if n % 2 == 0 else n

    colors = range(delta)
    edge_list = sorted({tuple(sorted((u, v))) for u, v in G.edges()})

    prob = pulp.LpProblem("EdgeColoring", pulp.LpMinimize)
    prob += 0

    x = {
        (edge, color): pulp.LpVariable(f"x_{edge[0]}_{edge[1]}_{color}", cat="Binary")
        for edge in edge_list
        for color in colors
    }

    for edge in edge_list:
        prob += pulp.lpSum(x[(edge, color)] for color in colors) == 1, f"EdgeColor_{edge}"

    for vertex in G.nodes():
        incident = [edge for edge in edge_list if vertex in edge]
        for color in colors:
            constraint = pulp.lpSum(x[(edge, color)] for edge in incident) <= 1
            prob += constraint, f"Vertex_{vertex}_Color_{color}"

    solver = _cbc_solver()
    prob.solve(solver)

    return delta if pulp.LpStatus[prob.status] == "Optimal" else delta + 1


def greedy_coloring_bound(G):
    """Return an upper bound on the chromatic number via greedy colouring."""

    if G.number_of_nodes() == 0:
        return 0
    coloring = nx.coloring.greedy_color(G, strategy="largest_first")
    return max(coloring.values()) + 1 if coloring else 0


def chromatic_number(G):
    """Return the chromatic number of ``G`` using a MILP formulation in PuLP."""

    n = G.number_of_nodes()
    if n == 0:
        return 0
    if is_complete(G):
        return n

    K = max(degree for _, degree in G.degree()) + 1
    nodes = list(G.nodes())
    colors = range(K)

    problem = pulp.LpProblem("ChromaticNumber", pulp.LpMinimize)

    # Variables: x[v][k] = 1 if vertex v has colour k, y[k] toggles colour usage.
    x = pulp.LpVariable.dicts("x", (nodes, colors), cat='Binary')
    y = pulp.LpVariable.dicts("y", colors, cat='Binary')

    problem += pulp.lpSum(y[k] for k in colors), "MinimizeColorCount"

    # Each vertex receives exactly one colour.
    for v in nodes:
        problem += pulp.lpSum(x[v][k] for k in colors) == 1, f"UniqueColor_{v}"
        # Adjacent vertices cannot share a colour.
        for u in G.neighbors(v):
            if str(v) < str(u):  # prevent handling each edge twice
                for k in colors:
                    problem += x[v][k] + x[u][k] <= 1, f"Adjacency_{v}_{u}_color_{k}"

    # Link assignment to colour usage indicator.
    for v in nodes:
        for k in colors:
            problem += x[v][k] <= y[k], f"UsageLink_{v}_{k}"

    solver = _cbc_solver()
    problem.solve(solver)
    return int(pulp.value(problem.objective))


def clique_number(G):
    """Return the size of a maximum clique in ``G`` (via ``nx.find_cliques``)."""

    cliques = list(nx.find_cliques(G))
    return max((len(clique) for clique in cliques), default=0)


def triangle_number(G):
    """Return the total number of triangles in ``G``."""

    tri = nx.triangles(G)
    return sum(tri.values()) // 3


def domination_number(G):
    """Return the domination number of ``G`` via a PuLP MILP model."""

    if G.number_of_nodes() == 0:
        return 0
    nodes = list(G.nodes())
    prob = pulp.LpProblem("DominationNumber", pulp.LpMinimize)
    x = {v: pulp.LpVariable(f"x_{v}", cat=pulp.LpBinary) for v in nodes}
    prob += pulp.lpSum(x[v] for v in nodes)
    for u in nodes:
        neighbors = list(G.neighbors(u)) + [u]
        prob += pulp.lpSum(x[v] for v in neighbors) >= 1
    prob.solve(_cbc_solver())
    return int(pulp.value(prob.objective))


def total_domination_number(G):
    """Return the total domination number of ``G`` via a PuLP MILP model."""

    if G.number_of_nodes() == 0:
        return 0
    nodes = list(G.nodes())
    prob = pulp.LpProblem("TotalDominationNumber", pulp.LpMinimize)
    x = {v: pulp.LpVariable(f"x_{v}", cat=pulp.LpBinary) for v in nodes}
    prob += pulp.lpSum(x[v] for v in nodes)
    for u in nodes:
        neighbors = list(G.neighbors(u))
        if neighbors:
            prob += pulp.lpSum(x[v] for v in neighbors) >= 1
        else:
            prob += x[u] == 1
    prob.solve(_cbc_solver())
    return int(pulp.value(prob.objective))


def independence_number(G):
    """Return the independence number of ``G`` via a PuLP MILP model."""

    if G.number_of_nodes() == 0:
        return 0
    nodes = list(G.nodes())
    prob = pulp.LpProblem("IndependenceNumber", pulp.LpMaximize)
    x = {v: pulp.LpVariable(f"x_{v}", cat=pulp.LpBinary) for v in nodes}
    prob += pulp.lpSum(x[v] for v in nodes)
    for u, v in G.edges():
        prob += x[u] + x[v] <= 1
    prob.solve(_cbc_solver())
    return int(pulp.value(prob.objective))


def vertex_cover_number(G):
    """Return the vertex cover number of ``G`` via a PuLP MILP model."""

    if G.number_of_nodes() == 0:
        return 0
    nodes = list(G.nodes())
    prob = pulp.LpProblem("VertexCoverNumber", pulp.LpMinimize)
    x = {v: pulp.LpVariable(f"x_{v}", cat=pulp.LpBinary) for v in nodes}
    prob += pulp.lpSum(x[v] for v in nodes)
    for u, v in G.edges():
        prob += x[u] + x[v] >= 1
    prob.solve(_cbc_solver())
    return int(pulp.value(prob.objective))


def independent_domination_number(G):
    """Return the independent domination number of ``G`` via a PuLP MILP model."""

    if G.number_of_nodes() == 0:
        return 0
    nodes = list(G.nodes())
    prob = pulp.LpProblem("IndependentDominationNumber", pulp.LpMinimize)
    x = {v: pulp.LpVariable(f"x_{v}", cat=pulp.LpBinary) for v in nodes}
    prob += pulp.lpSum(x[v] for v in nodes)
    # Independence constraint
    for u, v in G.edges():
        prob += x[u] + x[v] <= 1
    # Domination constraint (closed neighbourhood)
    for u in nodes:
        neighbors = list(G.neighbors(u)) + [u]
        prob += pulp.lpSum(x[v] for v in neighbors) >= 1
    prob.solve(_cbc_solver())
    return int(pulp.value(prob.objective))


def matching_number(G):
    """Return the size of a maximum matching in ``G``."""

    matching = nx.max_weight_matching(G, maxcardinality=True)
    return len(matching)


def feedback_vertex_set_number(G):
    """
    Calcule la taille minimale exacte du Feedback Vertex Set (FVS) de G.
    Utilise une approche itérative (Constraint Generation) avec PuLP.

    Returns:
        int: La taille du FVS (nombre de sommets à supprimer pour obtenir une forêt).
    """
    if G.number_of_nodes() == 0:
        return 0

    H = G.to_undirected()
    nodes = list(H.nodes())

    # On veut minimiser la somme des x_v (où x_v = 1 signifie "sommet supprimé")
    prob = pulp.LpProblem("FeedbackVertexSet_Exact", pulp.LpMinimize)
    x = {v: pulp.LpVariable(f"x_{v}", cat=pulp.LpBinary) for v in nodes}

    prob += pulp.lpSum(x[v] for v in nodes)

    solver = pulp.PULP_CBC_CMD(msg=0)

    while True:
        prob.solve(solver)

        # Reconstruire le graphe induit par les sommets GARDÉS (x_v = 0)
        kept_nodes = [v for v in nodes if pulp.value(x[v]) < 0.5]
        subgraph = H.subgraph(kept_nodes)

        try:
            # nx.find_cycle lève une exception s'il n'y a pas de cycle
            cycle_edges = nx.find_cycle(subgraph)

            cycle_nodes = set()
            for u, v in cycle_edges:
                cycle_nodes.add(u)
                cycle_nodes.add(v)

            prob += pulp.lpSum(x[v] for v in cycle_nodes) >= 1

        except nx.NetworkXNoCycle:
            return int(pulp.value(prob.objective))


def spanning_tree_number(G):
    """Return the number of spanning trees in ``G`` via Kirchhoff's theorem."""

    if G.number_of_nodes() == 0:
        return 0
    if not nx.is_connected(G):
        return 0
    adjacency = nx.to_numpy_array(G, dtype=float)
    degrees = np.diag(np.sum(adjacency, axis=1))
    laplacian = degrees - adjacency
    laplacian_minor = laplacian[1:, 1:]
    det = np.linalg.det(laplacian_minor)
    return int(round(abs(det)))


def vertex_connectivity(G):
    """Return the vertex connectivity of ``G``."""

    return nx.node_connectivity(G)


def edge_connectivity(G):
    """Return the edge connectivity of ``G``."""

    return nx.edge_connectivity(G)




def proximity(G):
    """Return the proximity of a connected graph ``G``."""

    n = G.number_of_nodes()
    if n == 0:
        raise ValueError("Proximity is undefined for the empty graph.")
    if not nx.is_connected(G):
        raise ValueError("Graph must be connected to compute proximity.")
    if n == 1:
        return 0.0

    distances = dict(nx.all_pairs_shortest_path_length(G))
    avg_distances = {
        node: sum(lengths.values()) / (n - 1)
        for node, lengths in distances.items()
    }
    return min(avg_distances.values())


def remoteness(G):
    """Return the remoteness of a connected graph ``G``."""

    n = G.number_of_nodes()
    if n == 0:
        raise ValueError("Remoteness is undefined for the empty graph.")
    if not nx.is_connected(G):
        raise ValueError("Graph must be connected to compute remoteness.")
    if n == 1:
        return 0.0

    distances = dict(nx.all_pairs_shortest_path_length(G))
    avg_distances = {
        node: sum(lengths.values()) / (n - 1)
        for node, lengths in distances.items()
    }
    return max(avg_distances.values())


def harmonic_index(G):
    """Return the harmonic index ``sum_{uv in E} 2 / (deg(u) + deg(v))``."""

    total = 0.0
    for u, v in G.edges():
        denominator = G.degree(u) + G.degree(v)
        if denominator:
            total += 2.0 / denominator
    return total


def randic_index(G):
    """Return the Randić index ``sum 1 / sqrt(deg(u) * deg(v))``."""

    total = 0.0
    for u, v in G.edges():
        prod = G.degree(u) * G.degree(v)
        if prod > 0:
            total += 1.0 / math.sqrt(prod)
    return total


def get_adjacency_matrix(G):
    """Return the adjacency matrix and the corresponding sorted node list."""

    nodes = get_sorted_nodes(G)
    return nx.to_numpy_array(G, nodelist=nodes), nodes


def get_distance_matrix(G):
    """Return the pairwise distance matrix (and node list) for a connected graph."""

    nodes = get_sorted_nodes(G)
    dist = dict(nx.all_pairs_shortest_path_length(G))
    size = len(nodes)
    matrix = np.zeros((size, size))
    for i, u in enumerate(nodes):
        for j, v in enumerate(nodes):
            matrix[i, j] = dist[u][v]
    return matrix, nodes


def companion_row(matrix: np.ndarray):
    """Return the last row of the companion matrix derived from ``matrix``."""

    degree = matrix.shape[0]
    coefficients = np.poly(matrix)
    row = [-coefficients[degree - idx] for idx in range(degree)]
    row.append(1)
    return row


def kth_largest_distance_eigenvalue(G, k):
    """Return the k-th largest eigenvalue of the distance matrix (1-indexed)."""

    if k < 1:
        return None
    matrix, _ = get_distance_matrix(G)
    eigenvalues = np.sort(np.real(np.linalg.eigvals(matrix)))[::-1]
    if k - 1 >= len(eigenvalues):
        return None
    return float(eigenvalues[k - 1])


def modified_zagreb_2(G):
    """Return the modified second Zagreb index ``sum 1/(deg(u)deg(v))``."""

    total = 0.0
    for u, v in G.edges():
        prod = G.degree(u) * G.degree(v)
        if prod:
            total += 1.0 / prod
    return total


def p_A(G):
    """Return the position of the largest absolute coefficient in the adjacency companion row."""

    adjacency, _ = get_adjacency_matrix(G)
    row = companion_row(adjacency)
    nonzero = [abs(value) for value in row if value]
    if not nonzero:
        return 0
    index = int(np.argmax(nonzero))
    return index + 1


def p_D(G):
    """Return the position of the maximal normalised coefficient from the distance matrix companion row."""

    n = G.number_of_nodes()
    distance_matrix, _ = get_distance_matrix(G)
    row = companion_row(distance_matrix)
    weights = [abs(row[idx]) * (2 ** (idx + 2 - n)) for idx in range(len(row))]
    return int(np.argmax(weights))


def m(G):
    """Return the number of non-zero coefficients in the companion row of the adjacency matrix."""

    adjacency, _ = get_adjacency_matrix(G)
    row = companion_row(adjacency)
    return sum(1 for value in row if not np.isclose(value, 0.0))


def connectivity(G):
    """Return the algebraic connectivity (second smallest Laplacian eigenvalue)."""

    if G.number_of_nodes() < 2:
        return 0.0
    laplacian = nx.laplacian_matrix(G).toarray()
    eigenvalues = np.sort(np.real(np.linalg.eigvals(laplacian)))
    return float(eigenvalues[1])


def spectrum(G):
    """Return the adjacency spectrum as real numbers."""

    adjacency, _ = get_adjacency_matrix(G)
    eigenvalues = np.linalg.eigvals(adjacency)
    return np.real(eigenvalues)


def largest_eigenvalue(G):
    """Return the largest eigenvalue of the adjacency matrix of ``G``."""

    if G.number_of_nodes() == 0:
        return 0.0
    adjacency = nx.to_numpy_array(G, dtype=float)
    eigenvalues = np.linalg.eigvals(adjacency)
    return float(np.max(eigenvalues.real))


def largest_distance_eigenvalue(G):
    """Return the largest eigenvalue of the distance matrix of ``G``."""

    if G.number_of_nodes() == 0:
        return 0.0
    if not nx.is_connected(G):
        return float("inf")
    nodes = list(G.nodes())
    n = len(nodes)
    dist_matrix = np.zeros((n, n))
    lengths = dict(nx.all_pairs_shortest_path_length(G))
    for i, u in enumerate(nodes):
        for j, v in enumerate(nodes):
            dist_matrix[i, j] = lengths[u][v]
    eigenvalues = np.linalg.eigvals(dist_matrix)
    return float(np.max(eigenvalues.real))


def second_smallest_laplace_eigenvalue(G):
    """Return the algebraic connectivity (second smallest Laplacian eigenvalue)."""

    if G.number_of_nodes() < 2:
        return 0.0
    adjacency = nx.to_numpy_array(G, dtype=float)
    degrees = np.diag(np.sum(adjacency, axis=1))
    laplacian = degrees - adjacency
    eigenvalues = np.linalg.eigvals(laplacian)
    eigenvalues = np.sort(eigenvalues.real)
    return float(eigenvalues[1])

def compute_invariants(G: nx.Graph, np_hard_invariants: bool) -> Dict[str, float]:
    """Calcule les propriétés topologiques pour guider le LLM."""

    list_invariants = {
        "is_connected": is_connected(G),
        "is_complete": is_complete(G),
        "is_tree": is_tree(G),
        "is_path": is_path(G),
        "is_star": is_star(G),
        "is_planar": is_planar(G),
        "is_chordal": is_chordal(G),
        "is_bipartite": is_bipartite(G),
        "is_triangle_free": triangle_number(G),
        "is_eulerian": is_eulerian(G),
        "is_hamiltonian": is_hamiltonian(G),
        "is_regular": is_regular(G),
        "contains_induced_subgraph": contains_induced_subgraph(G),
        "is_claw_free": is_claw_free(G),
        "is_bull_free": is_bull_free(G),
        "is_paw_free": is_paw_free(G),
        "is_diamond_free": is_diamond_free(G),
        "diameter": diameter(G),
        "radius": radius(G),

        "number_of_components": number_of_components(G),
        "largest_component_ratio": largest_component_ratio(G),
        "degree_variance": degree_variance(G),
        "average_clustering": average_clustering(G),
        # "number_of_leaves": number_of_leaves(G),
        "number_of_articulation_points": number_of_articulation_points(G),
        "number_of_bridges": number_of_bridges(G),
        # "girth": girth(G),
        # "circumference": circumference(G),
        "size": size(G),
        "order": order(G),
        "max_degree": maximum_degree(G),
        "min_degree": minimum_degree(G),
        "avg_degree": average_degree(G),
        "density": density(G),
        # "matching_number": matching_number(G),
        # "spanning_tree_number": spanning_tree_number(G),
        # "vertex_connectivity": vertex_connectivity(G),
        # "edge_connectivity": edge_connectivity(G),
        "triangle_number": triangle_number(G),
        # "proximity": proximity(G),
        # "remoteness": remoteness(G),
        "harmonic_index": harmonic_index(G),
        "randic_index": randic_index(G),
        "modified_zagreb_2": modified_zagreb_2(G),
        # "spectrum": spectrum(G),
        # "largest_eigenvalue": largest_eigenvalue(G),
        # "largest_distance_eigenvalue": largest_distance_eigenvalue(G),
        "connectivity": connectivity(G),
        # "p_A": p_A(G),
        # "p_D": p_D(G),
        # "m": m(G)
    }

    # if np_hard_invariants:
    #     list_update(
    #         {
    #             "treewidth": treewidth(G),
    #             "longest_path": longest_path(G),
    #             "longest_induced_path": longest_induced_path(G),
    #             "longest_induced_cycle": longest_induced_cycle(G),
    #             "chromatic_number": chromatic_number(G),
    #             "chromatic_index": chromatic_index(G),
    #             "clique_number": clique_number(G),
    #             "independence_number": independence_number(G),
    #             "vertex_cover_number": vertex_cover_number(G),
    #             "feedback_vertex_set_number": feedback_vertex_set_number(G),
    #             "domination_number": domination_number(G),
    #             "total_domination_number": total_domination_number(G),
    #             "independent_domination_number": independent_domination_number(G)
    #         }
    #     )

    return list_invariants

binary_properties_functions = {
    "connected": is_connected,
    "complete": is_complete,
    "tree": is_tree,
    "path": is_path,
    "star": is_star,
    "planar": is_planar,
    "chordal": is_chordal,
    "bipartite": is_bipartite,
    "triangle_free": is_triangle_free,
    "eulerian": is_eulerian,
    "hamiltonian": is_hamiltonian,
    "regular": is_regular,
    "claw_free": is_claw_free,
    "bull_free": is_bull_free,
    "paw_free": is_paw_free,
    "diamond_free": is_diamond_free,
}