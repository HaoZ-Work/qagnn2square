import networkx as nx
DG = nx.DiGraph()
DG.add_edge(2, 1,weight=1.0)   # adds the nodes in order 2, 1
DG.add_edge(1, 3,weight=1.0)
DG.add_edge(2, 4,weight=1.0)
DG.add_edge(1, 2,weight=1.0)

print(DG[2][1])
