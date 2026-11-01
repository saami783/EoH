import argparse
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("graph", type=str)
args = parser.parse_args()

g6_string = args.graph.encode("ascii")
nom_figure = Path("out/graph_generated.png")
nom_figure.parent.mkdir(parents=True, exist_ok=True)

G = nx.from_graph6_bytes(g6_string)

nb_sommets = G.number_of_nodes()
nb_aretes = G.number_of_edges()

with open("stats_graphes.txt", "a", encoding="utf-8") as fichier:
    fichier.write(f"{nom_figure} : Sommets : {nb_sommets} Arêtes : {nb_aretes}\n")

pos = nx.spring_layout(G, seed=42)

options = {
    'node_color': 'white',
    'edgecolors': 'dimgray',
    'linewidths': 1.5,
    'node_size': 150,
    'edge_color': 'darkgray',
    'width': 1.0,
    'with_labels': False
}

plt.figure(figsize=(6, 6))
nx.draw(G, pos, **options)

plt.axis('off')
plt.savefig(nom_figure, dpi=300, bbox_inches="tight", pad_inches=0.02)