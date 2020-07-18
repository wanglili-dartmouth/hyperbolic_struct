import networkx as nx
graph = nx.read_weighted_edgelist("usa-airports.edgelist", delimiter=" ", nodetype=int,create_using=nx.Graph())
graph=nx.convert_node_labels_to_integers(graph,first_label=0)
nx.write_edgelist(graph, "usa-airports.edgelist2", delimiter=" ", data=["weight"])
