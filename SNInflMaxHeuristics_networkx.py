# -*- coding: utf-8 -*-

import networkx as nx
import heapq as hq
import heuristics.SNSigmaSim_networkx as SNSim
import os
from src_MTO import saveload
from src_problem import util, spread

import time

# (Kempe) "The high-degree heuristic chooses nodes v in order of decreasing degrees. 
# Considering high-degree nodes as influential has long been a standard approach 
# for social and other networks [3, 83], and is known in the sociology literature 
# as 'degree centrality'."
# This code works also for directed graphs; assumes the edges point OUT of the influencer,
# e.g., "A influences B", "A is followed by B", "A is trusted by B".
# -> Calculates the k nodes of highest degree
def high_degree_nodes(k, G):

	if nx.is_directed(G):
		my_degree_function = G.out_degree
	else:
		my_degree_function = G.degree

	# the list of nodes to be returned; initialization
	H = [(my_degree_function(i), i) for i in list(G)[0:k]]
	hq.heapify(H) # min-heap

	for i in list(G)[k:]: # iterate through the remaining nodes
		deg_i = my_degree_function(i)
		if deg_i > H[0][0]:
			hq.heappushpop(H, (deg_i, i))

	return list(map(lambda x: x[1], H))

# (Kempe) Greedily adds to the set $S$ select nodes in order of increasing average distance 
# to other nodes in the network; following the intuition that being able to reach other nodes 
# quickly translates into high influence. distance = |V| for disconnected node pairs.
# This code works also for directed graphs; assumes the edges point OUT of the influencer,
# e.g., "A influences B", "A is followed by B", "A is trusted by B".
# -> Calculates the k nodes of lowest average distance to the other nodes
def low_distance_nodes(k, G):

    path_lens = nx.all_pairs_shortest_path_length(G) # contains only existing path lengths

    max_path_len = G.size()
    L = []

    for n in G.nodes():
        # compute the average distance per node
        avg_path_len_n = max_path_len

        if n in path_lens:
            sum_path_len_n = 0
            for m in set(G.nodes()) - set([n]):
                if m in path_lens[n]:
                    sum_path_len_n += path_lens[n][m]
                else:
                    sum_path_len_n += max_path_len
            avg_path_len_n = sum_path_len_n / (G.size() - 1)

        # add the average distance of n to L
        L.append((-avg_path_len_n, n)) # negated distance, to match the min-heap

    # L.sort(reverse=True) # expensive, so heap below
    H = L[0:k] 
    hq.heapify(H) # min-heap

    for i in L[k:]: # iterate through the remaining nodes
        if i[0] > H[0][0]:
            hq.heappushpop(H, i)

    return list(map(lambda x: (-x[0], x[1]), H))

# The SingleDiscount algorithm by Chen et al. (KDD'09) for any cascade model.
# This code works also for directed graphs; assumes the edges point OUT of the influencer,
# e.g., "A influences B", "A is followed by B", "A is trusted by B".
# -> Calculates the k nodes of highest degree, making discounts if direct neighbours are already chosen.
def single_discount_high_degree_nodes(k, G):
    if nx.is_directed(G):
        my_predecessor_function = G.predecessors
        my_degree_function = G.out_degree
    else:
        my_predecessor_function = G.neighbors
        my_degree_function = G.degree

    S = []
    ND = {}
    for n in G.nodes():
        ND[n] = my_degree_function(n)

    for i in range(k):
        # find the node of max degree not already in S
        u = max(set(list(ND.keys())) - set(S), key=(lambda key: ND[key]))
        S.append(u)

        # discount out-edges to u from all other nodes
        for v in my_predecessor_function(u):
            ND[v] -= 1

    return S

# Generalized Degree Discount from Wang et al., PlosOne'16.
# Only designed for Independent Cascade (hence p is passed as an argument) and undirected graphs.
# This code works also for directed graphs; assumes the edges point OUT of the influencer,
# e.g., "A influences B", "A is followed by B", "A is trusted by B".
# -> Calculates the k nodes of highest degree, making discounts if neighbours up to some depth are already chosen.
def generalized_degree_discount(k, G, p):
    if nx.is_directed(G):
        my_predecessor_function = G.predecessors
        my_degree_function = G.out_degree
    else:
        my_predecessor_function = G.neighbors
        my_degree_function = G.degree

    S = []
    GDD = {}
    t = {}

    for n in G.nodes():
        GDD[n] = my_degree_function(n)
        t[n] = 0

    for i in range(k):
        # select the node with current max GDD from V-S
        u = max(set(list(GDD.keys())) - set(S), key=(lambda key: GDD[key]))
        S.append(u)
        NB = set()

        # find the nearest and next nearest neighbors of u and update tv for v in Γ(u)
        for v in my_predecessor_function(u):
            NB.add(v)
            t[v] += 1
            for w in my_predecessor_function(v):
                if w not in S:
                    NB.add(w)
        # update gddv for all v in NB
        for v in NB:
            sumtw = 0
            for w in my_predecessor_function(v):
                if w not in S:
                    sumtw = sumtw + t[w]
            dv = my_degree_function(v)
            GDD[v] = dv - 2*t[v] - (dv - t[v])*t[v]*p + 0.5*t[v]*(t[v] - 1)*p - sumtw*p
            if GDD[v] < 0:
                GDD[v] = 0

    return S

# The algorithm proven to approximate within 63% of the optimal by Kempe, et al. for any cascade model.
# Hugely expensive in time.
# -> Prints (rather than returns) the 1..k nodes of supposedly max influence, and that influence.
# (It gets too time-expensive otherwise.)
def general_greedy(k, G, p, no_simulations, model):
    S = []

    for i in range(k):
        maxinfl_i = (-1, -1)
        v_i = -1
        for v in list(set(G.nodes()) - set(S)):
            eval_tuple = SNSim.evaluate(G, S+[v], p, no_simulations, model)
            if eval_tuple[0] > maxinfl_i[0]:
                maxinfl_i = (eval_tuple[0], eval_tuple[2])
                v_i = v

        S.append(v_i)
        print(i+1, maxinfl_i[0], maxinfl_i[1], S)

    return S, maxinfl_i

# CELF (Leskovec, Cost-effective Outbreak Detection in Networks, KDD07) is proven to approximate within 63% of the optimal.
# -> Prints (rather than returns) the 1..k nodes of supposedly max influence, and that influence.
# (It gets too time-expensive otherwise.)
# -> Does return only the final set of exactly k nodes.
def CELF(k, G, p, no_simulations, model):
    A = []

    max_delta = len(G.nodes()) + 1
    delta = {}
    for v in G.nodes():
        delta[v] = max_delta
    curr = {}

    while len(A) < k:
        for j in set(G.nodes()) - set(A):
            curr[j] = False
        while True:
            # find the node s from V-A which maximizes delta[s]
            max_curr = -1
            s = -1
            for j in set(G.nodes()) - set(A):
                if delta[j] > max_curr:
                    max_curr = delta[j]
                    s = j
            # evaluate s only if curr = False
            if curr[s]:
                A.append(s)
                # the result for this seed set is:
                res = SNSim.evaluate(G, A, p, no_simulations, model)
                print(len(A), res[0], res[2], A, sep=' ') # mean, CI95, A
                break
            else:
                eval_after = SNSim.evaluate(G, A+[s], p, no_simulations, model)
                eval_before = SNSim.evaluate(G, A, p, no_simulations, model)
                delta[s] = eval_after[0] - eval_before[0]
                curr[s] = True

    return A

def dump_degree_list(G):
    H = []

    for i in G.nodes():
        H.append((i, G.out_degree(i)))

    return H

def build_new_G(G):
    nodes = []
    edges = []
    nodes_id = dict()
    nodes_label = dict()
    # edges_id = []
    for id, label in enumerate(G.nodes()):
        nodes_id[label] = id
        nodes_label[id] = label
        nodes.append(id)
    for (v0, v1) in G.edges():
        print(v1)
        temp = [nodes_id[v0], nodes_id[v1]]
        edges.append(temp)
    # edges_id = copy.deepcopy(edges)
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    return G

if __name__ == "__main__":
    filen = 'GN-network'  # Hamsterster：0.03 fb-pages-public-figure：0.04 facebook_combined：0.02 Email_URV：0.05
    # compute the seed sets for the upper bound for k
    KK = [3, 6, 9, 12, 15, 18, 21, 24, 27, 30]
    # KK = [9, 12, 15, 18, 21, 24, 27, 30]
    # KK = [27, 30]
    p = 0.02
    num_sims = 10000
    model = 'IC'
    file = 'Graphs/' + filen + '.txt' # A trusts B
    # G = util.read_undirected_graph(file)
    G = util.read_graph(file)
    G = build_new_G(G)
    print(nx.average_clustering(G))
    # G = util.read_graph(file)

    # file = 'wiki-Vote.txt' # A votes B
    # file = 'amazon0302.txt'
    # file = 'web-Google.txt'
    # file = 'CA-GrQc.txt'
    # tempG = nx.read_edgelist(file, comments='#', delimiter='\t', create_using=nx.DiGraph(), nodetype=int, data=False)
    # G = tempG.reverse() # to get the edges to flow OUT of the influencers

    #file = 'graphs/twitter_combined.txt' # A follows B
    #tempG = nx.read_edgelist(file, comments='#', delimiter=' ', create_using=nx.DiGraph(), nodetype=int, data=False)
    #G = tempG.reverse()

    # file = 'facebook_combined.txt'
    # G = nx.read_edgelist(file, comments='#', delimiter=' ', create_using=nx.Graph(), nodetype=int, data=False)

    # G = nx.grid_2d_graph(10, 5)

    print("Read graph:", len(list(G.nodes())), "nodes", len(list(G.edges())), "edges")

    resDEGREE = []
    resDEGREEnodes = []
    resDEGREEtime = []
    resSDD = []
    resSDDnodes = []
    resSDDtime = []
    resDIS = []
    resDISnodes = []
    resDIStime = []
    resCELF = []
    resCELFnodes = []
    resCELFtime = []
    respagerank = []
    respageranknodes = []
    respageranktime = []
    resGDD = []
    resGDDnodes = []
    resGDDtime = []
    resGA = []
    resGAnodes = []
    resGAtime = []

    for k in KK:
        # DEGREE
        # start = time.time()
        # A = high_degree_nodes(k, G)
        # end = time.time()
        # tt = end-start
        # resDEGREEtime.append(tt)
        # print("DEGREE_Times: ", tt)
        # res = spread.MonteCarlo_simulation(G, A, p, num_sims, model)
        # print("DEGREE_" + str(k) + ":", res)
        # resDEGREE.append(res)
        # resDEGREEnodes.append(A)

        # Single DD
        # start = time.time()
        # A = single_discount_high_degree_nodes(k, G)
        # end = time.time()
        # tt = end - start
        # resSDDtime.append(tt)
        # print("Single DD_Times: ", tt)
        # res = spread.MonteCarlo_simulation(G, A, p, num_sims, model)
        # print("Single DD_" + str(k) + ":", res)
        # resSDD.append(res)
        # resSDDnodes.append(A)
        #
        # # DISTANCE
        # start = time.time()
        # A = low_distance_nodes(k, G)
        # A.sort()
        # A = list(map(lambda x: x[1], A))
        # end = time.time()
        # tt = end - start
        # resDIStime.append(tt)
        # print("DISTANCE_Times: ", tt)
        # res = spread.MonteCarlo_simulation(G, A, p, num_sims, model)
        # print("DISTANCE_" + str(k) + ":", res)
        # resDIS.append(res)
        # resDISnodes.append(A)

        # CELF
        start = time.time()
        A = CELF(k, G, p, 1000, model)
        end = time.time()
        tt = end - start
        resCELFtime.append(tt)
        print("CELF_Times: ", tt)
        res = spread.MonteCarlo_simulation(G, A, p, num_sims, model)
        print("CELF_" + str(k) + ":", res)
        resCELF.append(res)
        resCELFnodes.append(A)

        # pagerank
        # start = time.time()
        # IM = nx.pagerank(G)
        # IM = sorted(IM.items(), key=lambda x: x[1], reverse=True)
        # A = [i[0] for i in IM]
        # A = A[:k]
        # end = time.time()
        # tt = end - start
        # respageranktime.append(tt)
        # print("pagerank_Times: ", tt)
        # res = spread.MonteCarlo_simulation(G, A, p, num_sims, model)
        # print("pagerank_" + str(k) + ":", res)
        # respagerank.append(res)
        # respageranknodes.append(A)
        #
        # # Generalized DD
        # start = time.time()
        # A = generalized_degree_discount(k, G, p)
        # end = time.time()
        # tt = end - start
        # resGDDtime.append(tt)
        # print("Generalized DD_Times: ", tt)
        # res = spread.MonteCarlo_simulation(G, A, p, num_sims, model)
        # print("Generalized DD_" + str(k) + ":", res)
        # resGDD.append(res)
        # resGDDnodes.append(A)

        # # GEN-GREEDY
        # start = time.time()
        # A, ress = general_greedy(k, G, p, num_sims, model) # this prints rather than returns
        # end = time.time()
        # tt = end - start
        # resGAtime.append(tt)
        # print("GA_Times: ", tt)
        # res = spread.MonteCarlo_simulation(G, A, p, num_sims, model)
        # print("GA_" + str(k) + ":", res)
        # resGA.append(res)
        # resGAnodes.append(A)

    path = 'D:/5.电脑备份/25号楼电脑备份/网络中的关键节点寻找问题/code/MTIM/Result/heuristics/' + filen
    if not os.path.exists(path):
        os.makedirs(path)
    # saveload.savelist(path + '/resDEGREE4' + '.txt', resDEGREE)
    # saveload.savelist(path + '/resDEGREEtime4' + '.txt', resDEGREEtime)
    # saveload.savelist(path + '/resDEGREEnodes4' + '.txt', resDEGREEnodes)
    # saveload.savelist(path + '/resSDD4' + '.txt', resSDD)
    # saveload.savelist(path + '/resSDDtime4' + '.txt', resSDDtime)
    # saveload.savelist(path + '/resSDDnodes4' + '.txt', resSDDnodes)
    # saveload.savelist(path + '/resDIS' + '.txt', resDIS)
    # saveload.savelist(path + '/resDIStime' + '.txt', resDIStime)
    # saveload.savelist(path + '/resDISnodes' + '.txt', resDISnodes)
    saveload.savelist(path + '/resCELF4' + '.txt', resCELF)
    saveload.savelist(path + '/resCELFtime4' + '.txt', resCELFtime)
    saveload.savelist(path + '/resCELFnodes4' + '.txt', resCELFnodes)
    # saveload.savelist(path + '/respagerank4' + '.txt', respagerank)
    # saveload.savelist(path + '/respageranktime4' + '.txt', respageranktime)
    # saveload.savelist(path + '/respageranknodes4' + '.txt', respageranknodes)
    # saveload.savelist(path + '/resGDD' + '.txt', resGDD)
    # saveload.savelist(path + '/resGDDtime' + '.txt', resGDDtime)
    # saveload.savelist(path + '/resGDDnodes' + '.txt', resGDDnodes)
    # saveload.savelist(path + '/resGA' + '.txt', resGA)
    # saveload.savelist(path + '/resGAtime' + '.txt', resGAtime)
    # saveload.savelist(path + '/resGAnodes' + '.txt', resGAnodes)

    # # evaluate the seed sets obtained by the heuristics above
    # for i in range(1, k+1):
    #    res = SNSim.evaluate(G, A[:i], p, num_sims, model)
    #    print(i, res[0], res[2], A[:i], sep=' ') # mean, CI95, A
