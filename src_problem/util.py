import networkx as nx

def read_undirected_graph(f):

	G = nx.read_edgelist(f, comments='#', delimiter=' ', create_using=nx.Graph(), nodetype=int, data=False)
	return G


def read_graph(filename, nodetype=int):

	graph_class = nx.DiGraph() # all graph files are directed
	G = nx.read_edgelist(filename, create_using=graph_class, nodetype=nodetype, data=False)

	# msg = ' '.join(["Read from file", filename, "the directed graph\n", nx.classes.function.info(G)])
	# logging.info(msg)

	return G