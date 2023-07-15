import geatpy as ea
import networkx as nx
import os
from src_problem import util, IMproblem, spread
from src_MTO import saveload, initdegree, MTEAIM


def runMTEAIM(filenameall, args, K, G, p, model, no_simulations):
    """===============================实例化问题对象==========================="""
    # G = util.read_undirected_graph("Graphs/" + args + ".txt")
    # G = util.read_graph("Graphs/" + args + ".txt")
    # p = 0.04
    # model = 'IC'
    # no_simulations = 20
    # K = 10
    problemEDV = IMproblem.IM_EDV(K, G, p)
    # problemMCMH = IMproblem.IM_MCMH(K, G, p, model, no_simulations)
    problemPS = IMproblem.IM_PS(K, G, p, model)
    problem = [problemEDV, problemPS]

    """=============================种群设置==================================="""
    Encoding = 'RI'  # 编码方式
    NINDs = [100 for _ in range(len(problem))]  # 种群规模
    population = []  # 创建种群列表
    for i in range(len(problem)):
        Field = ea.crtfld(Encoding, problem[i].varTypes, problem[i].ranges, problem[i].borders)  # 创建区域描述器
        population.append(ea.Population(Encoding, Field, NINDs[i]))  # 实例化种群对象（此时种群还没被初始化，仅仅是完成种群对象的实例化）
        """=============================种群初始化==================================="""

    population = initdegree.initdegree(population, G, K, NINDs)

    """===============================MTEAIM算法参数设置============================="""
    MTEAIMA = MTEAIM.MTEAIM(problem, population, filenameall)  # 实例化一个算法模板对象
    MTEAIMA.MAXGEN = [50 for _ in range(len(problem))]  # 最大进化代数
    MTEAIMA.trappedValue = [1e-6 for _ in range(len(problem))]  # “进化停滞”判断阈值
    MTEAIMA.maxTrappedCount = [50 for _ in range(len(problem))]  # 进化停滞计数器最大上限值，如果连续maxTrappedCount代被判定进化陷入停滞，则终止进化
    MTEAIMA.logTras = [1 for _ in range(len(problem))]  # 设置每隔多少代记录日志，若设置成0则表示不记录日志
    MTEAIMA.verbose = [True for _ in range(len(problem))]  # 设置是否打印输出日志信息
    MTEAIMA.drawing = [1 for _ in range(len(problem))]  # 设置绘图方式（0：不绘图；1：绘制结果图；2：绘制目标空间过程动画；3：绘制决策空间过程动画）

    """==========================调用算法模板进行种群进化========================"""
    ResultMTEAIM = MTEAIMA.run()  # 执行算法模板，得到最优个体以及最后一代种群

    """=================================MTEAIM输出结果=============================="""
    cm = []
    for i in range(len(problem)):
        ResultMTEAIM[i][0].save(filenameall+'Result_MTEAIM/Result' + str(i + 1))  # 把第i个任务最优个体的信息保存到文件中
        path = filenameall+'Result_log'
        if not os.path.exists(path):
            os.makedirs(path)
        saveload.save_dict(filenameall+'Result_log/log' + str(i + 1) + '.txt', MTEAIMA.log[i])

        print('第%s个任务：' % (i + 1))
        print('评价次数：%s' % MTEAIMA.evalsNum[i])
        print('时间已过 %s 秒' % MTEAIMA.passTime[i])
        if ResultMTEAIM[i][0].sizes != 0:
            print('最优的目标函数值为：%s' % (ResultMTEAIM[i][0].ObjV[0][0]))
            print('最优的控制变量值为：')
            for j in range(ResultMTEAIM[i][0].Phen.shape[1]):
                print(ResultMTEAIM[i][0].Phen[0, j])
        else:
            print('没找到可行解。')
        A = ResultMTEAIM[i][0].Phen[0]
        a = spread.approx_EDV(G, A, p)
        b = spread.MonteCarlo_simulation_max_hop(G, A, p, no_simulations, model)
        cm.append(a+b[0])
    ResultMTEAIM[cm.index(max(cm))][0].save(filenameall+'Result_MTEAIM/Result0')
    path = filenameall + 'Result_time'
    if not os.path.exists(path):
        os.makedirs(path)
    saveload.savelist(path + '/timeall.txt', MTEAIMA.passTime)

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

if __name__ == '__main__':
    timess = 1
    args = "GN-network" # Hamsterster：0.03 fb-pages-public-figure：0.04 facebook_combined：0.02 Email_URV：0.05 NetHept 0.05
    # KK = [3, 6, 9, 12, 15, 18, 21, 24, 27, 30]
    # KK = [21, 24, 27, 30]
    # KK = [9, 12, 15, 18, 21, 24, 27, 30]
    KK = [3, 12, 21, 30]
    # KK = [6, 9, 15, 18, 24, 27]
    # G = util.read_undirected_graph("Graphs/" + args + ".txt")
    G = util.read_graph("Graphs/" + args + ".txt")
    G = build_new_G(G)
    p = 0.05
    model = 'IC'
    no_simulations = 30

    for was in [0, 1, 2, 3, 4]:
        for K in KK:
            filenameall = 'Result/' + str(was) + '_MTEAIM/' + str(args) + '_EMP_' + str(K) + '/'
            runMTEAIM(filenameall, args, K, G, p, model, no_simulations)