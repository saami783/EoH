def conj_a(G):
    """Conjecture fictive : on veut tester y(G) <= x(G)

      avec x(G) = nombre de sommets (n) et y(G) = n + 5.
      L'écart f(G) = x(G) - y(G) vaudra toujours -5.0.
      C'est un contre-exemple immédiat garanti dès la première évaluation !
      """
    n = G.number_of_nodes()
    x_val = float(n)
    y_val = float(n + 5)

    return x_val - y_val