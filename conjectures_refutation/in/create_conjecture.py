i = 0
for i in range(1,103):
    with open(f"conjecture_{i}.py", "w", encoding="utf-8") as f:
        f.write(f"def conjecture_{i}(G): \n")
        f.write("   import networkx as nx\n")
        f.write("   from conjectures_refutation.helpers import invariants as inv\n")
        f.write("   ...")