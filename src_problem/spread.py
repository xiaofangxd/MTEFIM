import networkx as nx
import random
import numpy as np
import math
from src_problem import util
from src_MTO import saveload
import time
import os

""" Spread models"""

""" Simulations of spread for Indepenfent Cascade (IC) and Weighted Cascade (WC).
    Suits (un)directed graphs.
    Assumes the edges point OUT of the influencer, e.g., if A->B or A-B, then "A influences B".
    
"""

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

def IC_model (G, a, p, random_generator):
    """
    :param G: (un)directed graphs, networkx
    :param a: the set of initial active nodes, list
    :param p: the system-wide probability of influence on an edge in [0, 1]
    :param random_generator
    :return:
    """

    A = set(a) # the set of active nodes, init a
    B = set(a) # the set of nodes activated in the last completed iteration
    converged = False

    while not converged:
        nextB = set()
        for n in B:
            for m in set(G.neighbors(n)) - A:
                prob = random_generator.random()
                if prob <= p:
                    nextB.add(m)
        B = set(nextB)
        if not B:
            converged = True
        A |= B

    return len(A)

def WC_model (G, a, random_generator):
    """
    :param G: (un)directed graphs, networkx
    :param a: the set of initial active nodes, list
    :param random_generator
    :return:
    """

    A = set(a) # the set of active nodes, init a
    B = set(a) # the set of nodes activated in the last completed iteration
    converged = False

    if nx.is_directed(G):
        my_degree_function = G.in_degree
    else:
        my_degree_function = G.degree

    while not converged:
        nextB = set()
        for n in B:
            for m in set(G.neighbors(n)) - A:
                prob = random_generator.random()
                p = 1.0 / my_degree_function(m)
                if prob <= p:
                    nextB.add(m)
        B = set(nextB)
        if not B:
            converged = True
        A |= B

    return len(A)

def IC_model_max_hop (G, a, p, max_hop, random_generator):
    """
    :param G: (un)directed graphs, networkx
    :param a: the set of initial active nodes, list
    :param p: the system-wide probability of influence on an edge in [0, 1]
    :param maxhop
    :param random_generator
    :return:
    """

    A = set(a) # the set of active nodes, init a
    B = set(a) # the set of nodes activated in the last completed iteration
    converged = False

    while (not converged) and (max_hop > 0):
        nextB = set()
        for n in B:
            for m in set(G.neighbors(n)) - A:
                prob = random_generator.random()
                if prob <= p:
                    nextB.add(m)
        B = set(nextB)
        if not B:
            converged = True
        A |= B
        max_hop -= 1
    return len(A)

def WC_model_max_hop (G, a, max_hop, random_generator):
    """
    :param G: (un)directed graphs, networkx
    :param a: the set of initial active nodes, list
    :param maxhop
    :param random_generator
    :return:
    """

    A = set(a) # the set of active nodes, init a
    B = set(a) # the set of nodes activated in the last completed iteration
    converged = False

    if nx.is_directed(G):
        my_degree_function = G.in_degree
    else:
        my_degree_function = G.degree

    while (not converged) and (max_hop > 0):
        nextB = set()
        for n in B:
            for m in set(G.neighbors(n)) - A:
                prob = random_generator.random()
                p = 1.0 / my_degree_function(m)
                if prob <= p:
                    nextB.add(m)
        B = set(nextB)
        if not B:
            converged = True
        A |= B
        max_hop -= 1

    return len(A)

""" Evaluates a given seed set A, simulated "no_simulations" times.
	Returns a tuple: (the mean, the stdev).
"""

def MonteCarlo_simulation(G, A, p, no_simulations, model, random_generator=None):
    """

    :param G: (un)directed graphs, networkx
    :param A: the set of active nodes, list
    :param p: the system-wide probability of influence on an edge in [0, 1]
    :param no_simulations: the number of simulations
    :param model: IC/WC model
    :param random_generator
    :return: Influence, mean, std
    """

    if random_generator is None:
        random_generator = random.Random()
        random_generator.seed(next(iter(A))) # initialize random number generator with first seed in the seed set, to make experiment repeatable;

    results = []
    if model == 'WC':
        for i in range(no_simulations):
            results.append(WC_model(G, A, random_generator=random_generator))
    elif model == 'IC':
        for i in range(no_simulations):
            results.append(IC_model(G, A, p, random_generator=random_generator))

    return (np.mean(results), np.std(results))

def MonteCarlo_simulation_max_hop(G, A, p, no_simulations, model, max_hop=2, random_generator=None):
    """
    calculates approximated influence spread of a given seed set A, with
	information propagation limited to a maximum number of hops
	example: with max_hops = 2 only neighbours and neighbours of neighbours can be activated

    :param G: (un)directed graphs, networkx
    :param A: the set of active nodes, list
    :param p: the system-wide probability of influence on an edge in [0, 1]
    :param no_simulations: the number of simulations
    :param model: IC/WC model
    :param max_hop: maximum number of hops
    :param random_generator
    :return: Influence, mean, std
    """

    if random_generator is None:
        random_generator = random.Random()
        random_generator.seed(next(iter(A))) # initialize random number generator with first seed in the seed set, to make experiment repeatable;

    results = []
    if model == 'WC':
        for i in range(no_simulations):
            results.append(WC_model_max_hop(G, A, max_hop, random_generator=random_generator))
    elif model == 'IC':
        for i in range(no_simulations):
            results.append(IC_model_max_hop(G, A, p, max_hop, random_generator=random_generator))

    return (np.mean(results), np.std(results))

def approx_EDV(G, A, p):
    """
    Given a graph G with edge probabilities p, and a seed list A,
    computes a numerical approximation of the influence of A in G as the
    Expected Diffusion Value (EDV) from

    Jiang, Q., Song, G., Cong, G., Wang, Y., Si, W., Xie, K.:
    Simulated annealing based influence maximization in social networks. AAAI (2011)

    Known: It only approximates well under Independent Cascade propagation, and small p.
    :param G: (un)directed graphs, networkx
    :param A: the set of active nodes, list
    :param p: the system-wide probability of influence on an edge in [0, 1]
    :return: Influences
    """
    influence = len(A)
    neighbourhood = set()
    for v in A:
        neighbourhood |= set(G.neighbors(v))
    neighbourhood -= set(A)

    for v in neighbourhood:
        if nx.is_directed(G):
            rv = len(set(G.predecessors(v)) & set(A))  # you want predecessors (in-neighbours)
        else:
            rv = len(set(G.neighbors(v)) & set(A))
        influence += 1 - math.pow(1-p, rv)

    return influence

def repr_graph_PS(G, p):
    """
    Computes a deterministic representative graph of G with the Probability Sorting (PS) method from

    Parchas, P., Gullo, F., Papadias, D., Bonchi, F.:
    Uncertain graph processing through representative instances. ACM Trans. Database Syst. (2015)

    Only defined on undirected graphs.
    """

    def dis2(RG, G, v, p):
        return RG.degree(v) - int(round(p * G.degree(v))) # no mode=igraph.IN/OUT; undirected

    # at the end of this process, RG will have all nodes of G, but no edges
    RG = nx.Graph()
    RG.add_nodes_from(G.nodes(data=True))

    # all edges have the same probability p; no need for edge sorting

    # adding vertices to RG
    for e in G.edges:
        dis2_u = dis2(RG, G, e[0], p)
        dis2_v = dis2(RG, G, e[1], p)
        if abs(dis2_u + 1) + abs(dis2_v + 1) < abs(dis2_u) + abs(dis2_v):
            RG.add_edge(e[0], e[1])
    return RG

def approx_PS(RG, A, propagation_model):
    """
    Given a graph G with edge probabilities p, and a seed list A,
    computes a numerical approximation of the influence of A in G by simulating the propagation model
    once over a deterministic representative graph RG of G, computed with various heuristics (PS, ADR, ABM).
    """
    # RG = repr_graph_PS(G, p)

    res = MonteCarlo_simulation(RG, A, 1.00, 1, propagation_model, random_generator=None)
    return res[0]

if __name__ == '__main__':
    args = "GN-network" # Hamsterster：0.03 fb-pages-public-figure：0.04 facebook_combined：0.02 Email_URV：0.05 GN-network 0.05 NetHept 0.05
    p = 0.05
    # G = util.read_undirected_graph(args + ".txt")
    G = util.read_graph(args + ".txt")
    G = build_new_G(G)
    # print(G.nodes())
    # A = [1912, 107, 1367, 1810, 1467, 2630, 1791, 2244, 2108, 997]  # found with CELF, influence 284 (p=0.01)
    no_sim = 10000
    time_compare = []
    # KK = [3, 6, 9, 12, 15, 18, 21, 24, 27, 30]
    KK = [3, 12, 21, 30]
    # KK = [3]
    # for pp in [0, 1, 2]:
    #     resultmean = []
    #     resultstd = []
    #     for K in KK:
    #         A = np.loadtxt('F:/网络中的关键节点寻找问题/code/MTIM/Result/rSBGA/2_' + str(args) + '_' + str(K) + '/'+'Result_rSBGA/Result' + str(pp) + '/Phen.csv', delimiter = ',')
    #         r = MonteCarlo_simulation(G, A, p, no_sim, 'IC')
    #         print("Simulation_IC" + str(K) + ":", r)
    #         resultmean.append(r[0])
    #         resultstd.append(r[1])
    #     path = 'F:/网络中的关键节点寻找问题/code/MTIM/Result/rSBGA/2_' + args
    #     if not os.path.exists(path):
    #         os.makedirs(path)
    #     saveload.savelist(path + '/resultmean' + str(pp) + '.txt', resultmean)
    #     saveload.savelist(path + '/resultstd' + str(pp) + '.txt', resultstd)
    #
    # for pp in [0, 1, 2]:
    #     resultmean = []
    #     resultstd = []
    #     for K in KK:
    #         A = np.loadtxt('F:/网络中的关键节点寻找问题/code/MTIM/Result/rEMEA/2_' + str(args) + '_' + str(K) + '/'+'Result_rEMEA/Result' + str(pp) + '/Phen.csv', delimiter = ',')
    #         r = MonteCarlo_simulation(G, A, p, no_sim, 'IC')
    #         print("Simulation_IC" + str(K) + ":", r)
    #         resultmean.append(r[0])
    #         resultstd.append(r[1])
    #     path = 'F:/网络中的关键节点寻找问题/code/MTIM/Result/rEMEA/2_' + args
    #     if not os.path.exists(path):
    #         os.makedirs(path)
    #     saveload.savelist(path + '/resultmean' + str(pp) + '.txt', resultmean)
    #     saveload.savelist(path + '/resultstd' + str(pp) + '.txt', resultstd)
    #
    # for was in [2]:
    #     for pp in [0, 1, 2]:
    #         resultmean = []
    #         resultstd = []
    #         for K in KK: # D:\5.电脑备份\25号楼电脑备份\网络中的关键节点寻找问题\code\MTIM\Result\0_stGA\facebook_combined_3\Result_stGA\Result0
    #             A = np.loadtxt('D:/2.发表成果/11.网络中的关键节点寻找问题/code/MTIM/Result/' + str(was) + '_MTEAIML/' + str(args) + '_' + str(K) + '/Result_MTEAIML/Result' + str(pp) + '/Phen.csv', delimiter = ',')
    #             r = MonteCarlo_simulation(G, A, p, no_sim, 'IC')
    #             print("Simulation_IC" + str(K) + ":", r)
    #             resultmean.append(r[0])
    #             resultstd.append(r[1])
    #         path = 'D:/2.发表成果/11.网络中的关键节点寻找问题/code/MTIM/Result/' + str(was) + '_MTEAIML/' + args
    #         if not os.path.exists(path):
    #             os.makedirs(path)
    #         saveload.savelist(path + '/resultmean' + str(pp) + '.txt', resultmean)
    #         saveload.savelist(path + '/resultstd' + str(pp) + '.txt', resultstd)


    # for was in [0, 1, 2, 3, 4]:
    #     for pp in [0, 1, 2, 3]:
    #         resultmean = []
    #         resultstd = []
    #         for K in KK: # D:\5.电脑备份\25号楼电脑备份\网络中的关键节点寻找问题\code\MTIM\Result\0_stGA\facebook_combined_3\Result_stGA\Result0
    #             A = np.loadtxt('D:/2.发表成果/11.网络中的关键节点寻找问题/code/MTIM/Result/' + str(was) + '_stGA/' + str(args) + '_EMP_' + str(K) + '/Result_stGA/Result' + str(pp) + '/Phen.csv', delimiter = ',')
    #             r = MonteCarlo_simulation(G, A, p, no_sim, 'IC')
    #             print("Simulation_IC" + str(K) + ":", r)
    #             resultmean.append(r[0])
    #             resultstd.append(r[1])
    #         path = 'D:/2.发表成果/11.网络中的关键节点寻找问题/code/MTIM/Result/' + str(was) + '_stGA_EMP/' + args
    #         if not os.path.exists(path):
    #             os.makedirs(path)
    #         saveload.savelist(path + '/resultmean' + str(pp) + '.txt', resultmean)
    #         saveload.savelist(path + '/resultstd' + str(pp) + '.txt', resultstd)
    for was in [0, 1, 2, 3, 4]:
        for pp in [0, 1, 2, 3]:
            resultmean = []
            resultstd = []
            for K in KK: # D:\5.电脑备份\25号楼电脑备份\网络中的关键节点寻找问题\code\MTIM\Result\0_stGA\facebook_combined_3\Result_stGA\Result0
                A = np.loadtxt('D:/2.发表成果/11.网络中的关键节点寻找问题/code/MTIM/Result/' + str(was) + '_MTEAIML/' + str(args) + '_EMP_' + str(K) + '/Result_MTEAIML/Result' + str(pp) + '/Phen.csv', delimiter = ',')
                r = MonteCarlo_simulation(G, A, p, no_sim, 'IC')
                print("Simulation_IC" + str(K) + ":", r)
                resultmean.append(r[0])
                resultstd.append(r[1])
            path = 'D:/2.发表成果/11.网络中的关键节点寻找问题/code/MTIM/Result/' + str(was) + '_MTEAIML_EMP/' + args
            if not os.path.exists(path):
                os.makedirs(path)
            saveload.savelist(path + '/resultmean' + str(pp) + '.txt', resultmean)
            saveload.savelist(path + '/resultstd' + str(pp) + '.txt', resultstd)
    for was in [0, 1, 2, 3, 4]:
        for pp in [0, 1, 2, 3]:
            resultmean = []
            resultstd = []
            for K in KK: # D:\5.电脑备份\25号楼电脑备份\网络中的关键节点寻找问题\code\MTIM\Result\0_stGA\facebook_combined_3\Result_stGA\Result0
                A = np.loadtxt('D:/2.发表成果/11.网络中的关键节点寻找问题/code/MTIM/Result/' + str(was) + '_GMTEAIM/' + str(args) + '_EMP_' + str(K) + '/Result_MTEAIM/Result' + str(pp) + '/Phen.csv', delimiter = ',')
                r = MonteCarlo_simulation(G, A, p, no_sim, 'IC')
                print("Simulation_IC" + str(K) + ":", r)
                resultmean.append(r[0])
                resultstd.append(r[1])
            path = 'D:/2.发表成果/11.网络中的关键节点寻找问题/code/MTIM/Result/' + str(was) + '_GMTEAIM_EMP/' + args
            if not os.path.exists(path):
                os.makedirs(path)
            saveload.savelist(path + '/resultmean' + str(pp) + '.txt', resultmean)
            saveload.savelist(path + '/resultstd' + str(pp) + '.txt', resultstd)




    # for pp in [0, 1, 2]:
    #     resultmean = []
    #     resultstd = []
    #     for K in KK:
    #         A = np.loadtxt('D:/5.电脑备份/25号楼电脑备份/网络中的关键节点寻找问题/code/MTIM/Result/4_SBGA/' + str(args) + '_' + str(K) + '/'+'Result_SBGA/Result' + str(pp) + '/Phen.csv', delimiter = ',')
    #         r = MonteCarlo_simulation(G, A, p, no_sim, 'IC')
    #         print("Simulation_IC" + str(K) + ":", r)
    #         resultmean.append(r[0])
    #         resultstd.append(r[1])
    #     path = 'D:/5.电脑备份/25号楼电脑备份/网络中的关键节点寻找问题/code/MTIM/Result/4_SBGA/' + args
    #     if not os.path.exists(path):
    #         os.makedirs(path)
    #     saveload.savelist(path + '/resultmean' + str(pp) + '.txt', resultmean)
    #     saveload.savelist(path + '/resultstd' + str(pp) + '.txt', resultstd)
    #
    # for pp in [0, 1, 2]:
    #     resultmean = []
    #     resultstd = []
    #     for K in KK:
    #         A = np.loadtxt('D:/5.电脑备份/25号楼电脑备份/网络中的关键节点寻找问题/code/MTIM/Result/4_EMEA/' + str(args) + '_' + str(K) + '/'+'Result_EMEA/Result' + str(pp) + '/Phen.csv', delimiter = ',')
    #         r = MonteCarlo_simulation(G, A, p, no_sim, 'IC')
    #         print("Simulation_IC" + str(K) + ":", r)
    #         resultmean.append(r[0])
    #         resultstd.append(r[1])
    #     path = 'D:/5.电脑备份/25号楼电脑备份/网络中的关键节点寻找问题/code/MTIM/Result/4_EMEA/' + args
    #     if not os.path.exists(path):
    #         os.makedirs(path)
    #     saveload.savelist(path + '/resultmean' + str(pp) + '.txt', resultmean)
    #     saveload.savelist(path + '/resultstd' + str(pp) + '.txt', resultstd)
    #
    # for pp in [0, 1, 2]:
    #     resultmean = []
    #     resultstd = []
    #     for K in KK:
    #         A = np.loadtxt('F:/网络中的关键节点寻找问题/code/MTIM/Result/rstGA/2_' + str(args) + '_' + str(K) + '/'+'Result_rstGA/Result' + str(pp) + '/Phen.csv', delimiter = ',')
    #         r = MonteCarlo_simulation(G, A, p, no_sim, 'IC')
    #         print("Simulation_IC" + str(K) + ":", r)
    #         resultmean.append(r[0])
    #         resultstd.append(r[1])
    #     path = 'F:/网络中的关键节点寻找问题/code/MTIM/Result/rstGA/2_' + args
    #     if not os.path.exists(path):
    #         os.makedirs(path)
    #     saveload.savelist(path + '/resultmean' + str(pp) + '.txt', resultmean)
    #     saveload.savelist(path + '/resultstd' + str(pp) + '.txt', resultstd)

    # for pp in [0, 1, 2]:
    #     resultmean = []
    #     resultstd = []
    #     for K in KK:
    #         A = np.loadtxt('D:/5.电脑备份/25号楼电脑备份/网络中的关键节点寻找问题/code/MTIM/Result/MFEA/' + str(args) + '_' + str(K) + '/'+'Result_MFEA/Result' + str(pp) + '/Phen.csv', delimiter = ',')
    #         r = MonteCarlo_simulation(G, A, p, no_sim, 'IC')
    #         print("Simulation_IC" + str(K) + ":", r)
    #         resultmean.append(r[0])
    #         resultstd.append(r[1])
    #     path = 'D:/5.电脑备份/25号楼电脑备份/网络中的关键节点寻找问题/code/MTIM/Result/MFEA/' + args
    #     if not os.path.exists(path):
    #         os.makedirs(path)
    #     saveload.savelist(path + '/resultmean' + str(pp) + '.txt', resultmean)
    #     saveload.savelist(path + '/resultstd' + str(pp) + '.txt', resultstd)
    #
    # for pp in [0, 1, 2]:
    #     resultmean = []
    #     resultstd = []
    #     for K in KK:
    #         A = np.loadtxt('D:/5.电脑备份/25号楼电脑备份/网络中的关键节点寻找问题/code/MTIM/Result/rMFEA/' + str(args) + '_' + str(K) + '/'+'Result_rMFEA/Result' + str(pp) + '/Phen.csv', delimiter = ',')
    #         r = MonteCarlo_simulation(G, A, p, no_sim, 'IC')
    #         print("Simulation_IC" + str(K) + ":", r)
    #         resultmean.append(r[0])
    #         resultstd.append(r[1])
    #     path = 'D:/5.电脑备份/25号楼电脑备份/网络中的关键节点寻找问题/code/MTIM/Result/rMFEA/' + args
    #     if not os.path.exists(path):
    #         os.makedirs(path)
    #     saveload.savelist(path + '/resultmean' + str(pp) + '.txt', resultmean)
    #     saveload.savelist(path + '/resultstd' + str(pp) + '.txt', resultstd)
    #
    # for pp in [0, 1, 2]:
    #     resultmean = []
    #     resultstd = []
    #     for K in KK:
    #         A = np.loadtxt('D:/5.电脑备份/25号楼电脑备份/网络中的关键节点寻找问题/code/MTIM/Result/1_MFEAII/' + str(args) + '_' + str(K) + '/'+'Result_MFEAII/Result' + str(pp) + '/Phen.csv', delimiter = ',')
    #         r = MonteCarlo_simulation(G, A, p, no_sim, 'IC')
    #         print("Simulation_IC" + str(K) + ":", r)
    #         resultmean.append(r[0])
    #         resultstd.append(r[1])
    #     path = 'D:/5.电脑备份/25号楼电脑备份/网络中的关键节点寻找问题/code/MTIM/Result/1_MFEAII/' + args
    #     if not os.path.exists(path):
    #         os.makedirs(path)
    #     saveload.savelist(path + '/resultmean' + str(pp) + '.txt', resultmean)
    #     saveload.savelist(path + '/resultstd' + str(pp) + '.txt', resultstd)

    # for pp in [0, 1, 2]:
    #     resultmean = []
    #     resultstd = []
    #     for K in KK:
    #         A = np.loadtxt('D:/5.电脑备份/25号楼电脑备份/网络中的关键节点寻找问题/code/MTIM/Result/rMFEAII/' + str(args) + '_' + str(K) + '/'+'Result_rMFEAII/Result' + str(pp) + '/Phen.csv', delimiter = ',')
    #         r = MonteCarlo_simulation(G, A, p, no_sim, 'IC')
    #         print("Simulation_IC" + str(K) + ":", r)
    #         resultmean.append(r[0])
    #         resultstd.append(r[1])
    #     path = 'D:/5.电脑备份/25号楼电脑备份/网络中的关键节点寻找问题/code/MTIM/Result/rMFEAII/' + args
    #     if not os.path.exists(path):
    #         os.makedirs(path)
    #     saveload.savelist(path + '/resultmean' + str(pp) + '.txt', resultmean)
    #     saveload.savelist(path + '/resultstd' + str(pp) + '.txt', resultstd)

    # for was in [0, 1, 2, 3, 4]:
    #     for pp in [0, 1, 2]:
    #         resultmean = []
    #         resultstd = []
    #         for K in KK:
    #             A = np.loadtxt('D:/5.电脑备份/25号楼电脑备份/网络中的关键节点寻找问题/code/MTIM/Result/' + str(was) + '_GMTEAIM/' + str(args) + '_' + str(K) + '/'+'Result_MTEAIM/Result' + str(pp) + '/Phen.csv', delimiter = ',')
    #             r = MonteCarlo_simulation(G, A, p, no_sim, 'IC')
    #             print("Simulation_IC" + str(K) + ":", r)
    #             resultmean.append(r[0])
    #             resultstd.append(r[1])
    #         path = 'D:/5.电脑备份/25号楼电脑备份/网络中的关键节点寻找问题/code/MTIM/Result/' + str(was) + '_GMTEAIM/' + args
    #         if not os.path.exists(path):
    #             os.makedirs(path)
    #         saveload.savelist(path + '/resultmean' + str(pp) + '.txt', resultmean)
    #         saveload.savelist(path + '/resultstd' + str(pp) + '.txt', resultstd)

    # for pp in [0, 1, 2]:
    #     resultmean = []
    #     resultstd = []
    #     for K in KK:
    #         A = np.loadtxt('D:/5.电脑备份/25号楼电脑备份/网络中的关键节点寻找问题/code/MTIM/Result/rMTEAIM/' + str(args) + '_' + str(K) + '/'+'Result_rMTEAIM/Result' + str(pp) + '/Phen.csv', delimiter = ',')
    #         r = MonteCarlo_simulation(G, A, p, no_sim, 'IC')
    #         print("Simulation_IC" + str(K) + ":", r)
    #         resultmean.append(r[0])
    #         resultstd.append(r[1])
    #     path = 'D:/5.电脑备份/25号楼电脑备份/网络中的关键节点寻找问题/code/MTIM/Result/rMTEAIM/' + args
    #     if not os.path.exists(path):
    #         os.makedirs(path)
    #     saveload.savelist(path + '/resultmean' + str(pp) + '.txt', resultmean)
    #     saveload.savelist(path + '/resultstd' + str(pp) + '.txt', resultstd)


    # start = time.time()
    # print("Simulation_IC_2_hop:", MonteCarlo_simulation_max_hop(G, A, p, no_sim, 'IC'))
    # end = time.time()
    # time_compare.append(end - start)
    #
    # start = time.time()
    # print("Simulation_WC:", MonteCarlo_simulation(G, A, p, no_sim, 'WC'))
    # end = time.time()
    # time_compare.append(end - start)
    #
    # start = time.time()
    # print("Simulation_WC_2_hop:", MonteCarlo_simulation_max_hop(G, A, p, no_sim, 'WC'))
    # end = time.time()
    # time_compare.append(end - start)
    #
    # start = time.time()
    # print("EDV approximation:", approx_EDV(G, A, p))
    # end = time.time()
    # time_compare.append(end - start)
    # print("time of Simulation_IC:", time_compare[0])
    # print("time of Simulation_IC_2_hop:", time_compare[1])
    # print("time of Simulation_WC:", time_compare[2])
    # print("time of Simulation_WC_2_hop:", time_compare[3])
    # print("time of EDV approximation:", time_compare[4])
