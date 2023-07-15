import geatpy as ea
import numpy as np
from src_problem import spread


class IM_EDV(ea.Problem):
    def __init__(self, Dim, G, p):
        M = 1
        name = 'Influence Maximization EDV'
        maxormins = [-1] # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        varTypes = np.array([1]*Dim) # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        if 0 in G.nodes:
            self.lb = [0] * Dim  # 决策变量下界
            self.ub = [len(list(G.nodes))] * Dim  # 决策变量上界
            self.lbin = [1] * Dim  # 决策变量下边界
            self.ubin = [0] * Dim  # 决策变量上边界
        else:
            self.lb = [1] * Dim  # 决策变量下界
            self.ub = [len(list(G.nodes))] * Dim  # 决策变量上界
            self.lbin = [1] * Dim  # 决策变量下边界
            self.ubin = [1] * Dim  # 决策变量上边界
        self.G = G
        self.p = p
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, self.lb, self.ub, self.lbin, self.ubin)

    def aimFunc(self, pop):  # 目标函数
        Vars = pop.Phen  # 得到决策变量矩阵
        f = np.zeros((Vars.shape[0], 1))
        for i in range(Vars.shape[0]):
            A = set(list(Vars[i, :]))
            influence = spread.approx_EDV(self.G, A, self.p)
            f[i] = influence
        pop.ObjV = f  # 把求得的目标函数值赋值给种群pop的ObjV


class IM_MCMH(ea.Problem):
    def __init__(self, Dim, G, p, model, no_simulations):
        M = 1
        name = 'Influence Maximization MonteCarlo_simulation Max Hop'
        maxormins = [-1]  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        varTypes = np.array([1] * Dim)  # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        if 0 in G.nodes:
            self.lb = [0] * Dim  # 决策变量下界
            self.ub = [len(list(G.nodes))] * Dim  # 决策变量上界
            self.lbin = [1] * Dim  # 决策变量下边界
            self.ubin = [0] * Dim  # 决策变量上边界
        else:
            self.lb = [1] * Dim  # 决策变量下界
            self.ub = [len(list(G.nodes))] * Dim  # 决策变量上界
            self.lbin = [1] * Dim  # 决策变量下边界
            self.ubin = [1] * Dim  # 决策变量上边界
        self.G = G
        self.p = p
        self.model = model
        self.no_simulations = no_simulations
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, self.lb, self.ub, self.lbin, self.ubin)

    def aimFunc(self, pop):  # 目标函数
        Vars = pop.Phen  # 得到决策变量矩阵
        f = np.zeros((Vars.shape[0], 1))
        for i in range(Vars.shape[0]):
            A = set(list(Vars[i, :]))
            influence_mean, influence_std = spread.MonteCarlo_simulation_max_hop(self.G, A, self.p, self.no_simulations, self.model, max_hop=2)
            f[i] = influence_mean
        pop.ObjV = f  # 把求得的目标函数值赋值给种群pop的ObjV

class IM_PS(ea.Problem):
    def __init__(self, Dim, G, p, model, no_simulations=30):
        M = 1
        name = 'PS'
        maxormins = [-1]  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        varTypes = np.array([1] * Dim)  # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        if 0 in G.nodes:
            self.lb = [0] * Dim  # 决策变量下界
            self.ub = [len(list(G.nodes))] * Dim  # 决策变量上界
            self.lbin = [1] * Dim  # 决策变量下边界
            self.ubin = [0] * Dim  # 决策变量上边界
        else:
            self.lb = [1] * Dim  # 决策变量下界
            self.ub = [len(list(G.nodes))] * Dim  # 决策变量上界
            self.lbin = [1] * Dim  # 决策变量下边界
            self.ubin = [1] * Dim  # 决策变量上边界
        self.G = G
        self.p = p
        self.model = model
        self.no_simulations = no_simulations
        self.RG = spread.repr_graph_PS(self.G, self.p)
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, self.lb, self.ub, self.lbin, self.ubin)

    def aimFunc(self, pop):  # 目标函数
        Vars = pop.Phen  # 得到决策变量矩阵
        f = np.zeros((Vars.shape[0], 1))
        for i in range(Vars.shape[0]):
            A = set(list(Vars[i, :]))
            influence_mean = spread.approx_PS(self.RG, A, self.model)
            f[i] = influence_mean
        pop.ObjV = f  # 把求得的目标函数值赋值给种群pop的ObjV