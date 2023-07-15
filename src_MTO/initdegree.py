import numpy as np
import networkx as nx
from SNInflMaxHeuristics_networkx import generalized_degree_discount, single_discount_high_degree_nodes

def initGDD(population, G, K, NINDs, p):
    if nx.is_directed(G):
        my_degree_function = G.in_degree
    else:
        my_degree_function = G.degree
    degree = np.zeros(len(list(G.nodes)))
    if 0 in G.nodes:
        for i in range(len(list(G.nodes))):
            degree[i] = my_degree_function(i)
        degreesort = np.argsort(-degree)
    else:
        for i in range(len(list(G.nodes))):
            degree[i] = my_degree_function(i+1)
        degreesort = np.argsort(-degree) + 1
    for i in range(len(population)):
        Phen = np.zeros((NINDs[i], K))
        for j in range(NINDs[i]):
            Phen[j, :] = np.array(generalized_degree_discount(K, G, p))
            for k in range(K):
                if np.random.random() < 0.5:
                    Phen[j, k] = degreesort[np.random.randint(K, len(list(G.nodes)))]
        population[i].Phen = Phen
        population[i].Chrom = Phen
        population[i].Lind = population[i].Chrom.shape[1]  # 计算染色体的长度
        population[i].ObjV = None
        population[i].FitnV = None
        population[i].CV = None
    return population

def initSDD(population, G, K, NINDs, p):
    if nx.is_directed(G):
        my_degree_function = G.in_degree
    else:
        my_degree_function = G.degree
    degree = np.zeros(len(list(G.nodes)))
    if 0 in G.nodes:
        for i in range(len(list(G.nodes))):
            degree[i] = my_degree_function(i)
        degreesort = np.argsort(-degree)
    else:
        for i in range(len(list(G.nodes))):
            degree[i] = my_degree_function(i+1)
        degreesort = np.argsort(-degree) + 1
    for i in range(len(population)):
        Phen = np.zeros((NINDs[i], K))
        for j in range(NINDs[i]):
            Phen[j, :] = np.array(single_discount_high_degree_nodes(K, G))
            for k in range(K):
                if np.random.random() < 0.5:
                    Phen[j, k] = degreesort[np.random.randint(K, len(list(G.nodes)))]
        population[i].Phen = Phen
        population[i].Chrom = Phen
        population[i].Lind = population[i].Chrom.shape[1]  # 计算染色体的长度
        population[i].ObjV = None
        population[i].FitnV = None
        population[i].CV = None
    return population

def initpg(population, G, K, NINDs, p):
    if nx.is_directed(G):
        my_degree_function = G.in_degree
    else:
        my_degree_function = G.degree
    degree = np.zeros(len(list(G.nodes)))
    IM = nx.pagerank(G)
    IM = sorted(IM.items(), key=lambda x: x[1], reverse=True)
    A = [i[0] for i in IM]
    A = A[:K]
    if 0 in G.nodes:
        for i in range(len(list(G.nodes))):
            degree[i] = my_degree_function(i)
        degreesort = np.argsort(-degree)
    else:
        for i in range(len(list(G.nodes))):
            degree[i] = my_degree_function(i+1)
        degreesort = np.argsort(-degree) + 1
    for i in range(len(population)):
        Phen = np.zeros((NINDs[i], K))
        for j in range(NINDs[i]):
            Phen[j, :] = np.array(A)
            for k in range(K):
                if np.random.random() < 0.5:
                    Phen[j, k] = degreesort[np.random.randint(K, len(list(G.nodes)))]
        population[i].Phen = Phen
        population[i].Chrom = Phen
        population[i].Lind = population[i].Chrom.shape[1]  # 计算染色体的长度
        population[i].ObjV = None
        population[i].FitnV = None
        population[i].CV = None
    return population

def initdegree(population, G, K, NINDs):
    if nx.is_directed(G):
        my_degree_function = G.in_degree
    else:
        my_degree_function = G.degree
    degree = np.zeros(len(list(G.nodes)))
    if 0 in G.nodes:
        for i in range(len(list(G.nodes))):
            degree[i] = my_degree_function(i)
        degreesort = np.argsort(-degree)
    else:
        for i in range(len(list(G.nodes))):
            degree[i] = my_degree_function(i+1)
        degreesort = np.argsort(-degree) + 1
    for i in range(len(population)):
        Phen = np.zeros((NINDs[i], K))
        for j in range(NINDs[i]):
            Phen[j, :] = degreesort[0:K]
            for k in range(K):
                if np.random.random() < 0.5:
                    Phen[j, k] = degreesort[np.random.randint(K, len(list(G.nodes)))]
        population[i].Phen = Phen
        population[i].Chrom = Phen
        population[i].Lind = population[i].Chrom.shape[1]  # 计算染色体的长度
        population[i].ObjV = None
        population[i].FitnV = None
        population[i].CV = None
    return population