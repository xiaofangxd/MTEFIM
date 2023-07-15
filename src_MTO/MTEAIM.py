import geatpy as ea  # 导入geatpy库
import numpy as np
import random
from src_MTO.eaAlgorithm import MTOsogaAlgorithm


def repOper(N, oldChrom, G):
    """
    修复算子
    N : 图的节点个数
    """
    newChrom = oldChrom
    if 0 in G.nodes:
        a = np.arange(N)
    else:
        a = np.arange(N) + 1
    for i in range(newChrom.shape[0]):
        b, ind = np.unique(newChrom[i, :], return_index=True)
        if len(b) < newChrom.shape[1]:
            c = np.array(list(set(a) - set(b)))
            ind1 = list(set(np.arange(newChrom.shape[1])) - set(ind))
            newChrom[i, ind1] = c[np.random.randint(len(c), size=len(ind1))]

    return newChrom

def learnRMP(parent, G):

    numtasks = len(parent)
    rrec = []
    KK = parent[0].Chrom.shape[1] # KK是要找的解集大小
    for i in range(numtasks):
        temp = np.zeros((parent[i].Chrom.shape[0], len(list(G.nodes))))
        for j in range(parent[i].Chrom.shape[0]):
            if 0 in G.nodes:
                temp[j, parent[i].Chrom[j].astype(np.uint8)] = 1
            else:
                temp[j, (parent[i].Chrom[j]-1).astype(np.uint8)] = 1
        rrec.append(temp)

    RMP = np.eye(numtasks)
    maxdims = rrec[0].shape[1]

    for i in range(numtasks):
        for j in range(i+1, numtasks):
            pps = []
            for k in range(rrec[i].shape[0]):
                tmp = 0
                for d in range(rrec[j].shape[0]):
                    tmp = tmp + 2*KK - sum(rrec[i][k, :] != rrec[j][d, :])
                pps.append(tmp)

            RMP[i, j] = sum(pps)/(rrec[i].shape[0]*rrec[j].shape[0]*2*KK)
            RMP[j, i] = RMP[i, j]

    return RMP

def learnRMP2(parent, G):

    numtasks = len(parent)
    KK = parent[0].Chrom.shape[1] # KK是要找的解集大小

    RMP = np.eye(numtasks)

    for i in range(numtasks):
        for j in range(i+1, numtasks):
            pps = []
            for k in range(parent[i].Chrom.shape[0]):
                tmp = 0
                for d in range(parent[j].Chrom.shape[0]):
                    tmp = tmp + np.intersect1d(parent[i].Chrom[k, :], parent[j].Chrom[d, :]).size
                pps.append(tmp)

            RMP[i, j] = sum(pps)/(parent[i].Chrom.shape[0]*parent[j].Chrom.shape[0]*KK)
            RMP[j, i] = RMP[i, j]

    return RMP


class MTEAIM(MTOsogaAlgorithm):
    """
    MTEAIM: class -  解决影响力最大化的多任务优化算法
模板说明:
    该模板是解决影响力最大化的多任务优化算法
    注意：本算法模板中的problem和population为一个存储种群类对象的列表，而不是单个种群类对象。
算法描述:


"""

    def __init__(self, problem, population, filenameall):
        if type(population) != list or type(problem) != list:
            raise RuntimeError('传入的问题或者种群对象列表必须为list类型')
        MTOsogaAlgorithm.__init__(self, problem, population)  # 先调用父类构造方法
        self.filenameall = filenameall
        self.name = 'MTEAIM'
        self.PopNum = self.ProNum  # 种群数目
        self.migOpers = ea.Migrate(MIGR=0.2, Structure=2, Select=1, Replacement=2)  # 生成种群迁移算子对象
        # 为不同的种群设置不同的选择、重组、变异算子
        self.selFunc = ['tour' for _ in range(self.PopNum)]  # 锦标赛选择算子
        self.recOpers = []
        self.mutOpers = []
        self.RMP = 0.5*np.ones((self.ProNum, self.ProNum))
        Pms = []
        Pcs = []
        for i in range(self.PopNum):  # 遍历种群列表
            Pms.append(1 / self.problem[i].Dim)
            Pcs.append(1)
            pop = population[i]  # 得到当前种群对象
            if pop.Encoding == 'P':
                recOper = ea.Xovpmx(XOVR=Pcs[i])  # 生成部分匹配交叉算子对象
                mutOper = ea.Mutinv(Pm=float(Pms[i]))  # 生成逆转变异算子对象
            else:
                if pop.Encoding == 'BG':
                    recOper = ea.Xovdp(XOVR=Pcs[i])  # 生成两点交叉算子对象
                    mutOper = ea.Mutbin(Pm=float(Pms[i]))  # 生成二进制变异算子对象
                elif pop.Encoding == 'RI':
                    recOper = ea.Xovsp(XOVR=Pcs[i])  # 生成两点交叉算子对象
                    mutOper = ea.Mutuni(Pm=float(Pms[i]))
                    # mutOper = ea.Mutbga(Pm=float(Pms[i]), MutShrink=0.5, Gradient=20)  # 生成breeder GA变异算子对象
                else:
                    raise RuntimeError('编码方式必须为''BG''、''RI''或''P''.')
            self.recOpers.append(recOper)
            self.mutOpers.append(mutOper)

    def run(self, prophetPops=None):  # prophetPops为先知种群列表（即包含先验知识的种群列表）
        # ==========================初始化配置===========================
        self.initialization()  # 初始化算法模板的一些动态参数
        population = self.population  # 密切注意本模板的population是一个存储种群类对象的列表

        # ===========================准备进化============================
        for i in range(self.PopNum):  # 遍历每个种群，初始化每个种群的染色体矩阵
            # population[i].initChrom(population[i].sizes)  # 初始化种群染色体矩阵
            self.call_aimFunc(population[i], i)  # 计算种群的目标函数值
            # 插入先验知识（注意：这里不会对先知种群列表prophetPops的合法性进行检查）
            if prophetPops is not None:
                population[i] = (prophetPops[i] + population[i])[:population[i].sizes]  # 插入先知种群
            population[i].FitnV = ea.scaling(population[i].ObjV, population[i].CV, self.problem[i].maxormins)  # 计算适应度

        # ===========================开始进化============================
        rmp = [] # 记录rmp
        con = [False for _ in range(self.PopNum)]
        while False in con:
            for i in range(self.PopNum):
                con[i] = self.terminated(population[i], i)

            offspring = [None for _ in range(self.ProNum)]
            for i in range(self.PopNum):  # 遍历种群列表，分别对各个种群进行重组和变异
                if con[i] == True:
                    continue
                pop = population[i]  # 得到当前种群
                # 选择
                offspring[i] = pop[ea.selecting(self.selFunc[i], pop.FitnV, pop.sizes)]
                # 进行进化操作
                offspring[i].Chrom = self.recOpers[i].do(offspring[i].Chrom)  # 重组
                offspring[i].Chrom = repOper(offspring[i].Field[1, 0], offspring[i].Chrom, self.problem[i].G)
                offspring[i].Chrom = self.mutOpers[i].do(offspring[i].Encoding, offspring[i].Chrom, offspring[i].Field)  # 变异
                offspring[i].Chrom = repOper(offspring[i].Field[1, 0], offspring[i].Chrom, self.problem[i].G)

            # 更新RMP
            # self.RMP = learnRMP(population, self.problem[0].G) # 所有问题是同一个图,所以哪一个问题的图都可以
            self.RMP = learnRMP2(population, self.problem[0].G)  # 所有问题是同一个图,所以哪一个问题的图都可以
            print(self.RMP[0, 1])
            rmp.append(self.RMP[0, 1])

            for i in range(self.PopNum):  # 遍历种群列表，迁移个体并评价新种群
                if con[i] == True:
                    continue
                pp = self.RMP[i]
                pp = np.delete(pp, i)
                j = np.argmax(pp)
                if j >= i:
                    j = j+1
                if np.random.rand() < self.RMP[i][j] and offspring[j] != None:
                    s = int(np.floor(self.RMP[i][j] * offspring[0].Chrom.shape[0]))
                    ind1 = random.sample(range(offspring[i].Chrom.shape[0]), s)
                    ind2 = random.sample(range(offspring[j].Chrom.shape[0]), s)
                    offspring[i].Chrom[ind1, :] = offspring[j].Chrom[ind2, :]

                self.call_aimFunc(offspring[i], i)  # 计算目标函数值
                population[i] = population[i] + offspring[i]  # 父子合并
                population[i].FitnV = ea.scaling(population[i].ObjV, population[i].CV, self.problem[i].maxormins)  # 计算适应度
                population[i] = population[i][ea.selecting('dup', population[i].FitnV, pop.sizes)]  # 采用基于适应度排序的直接复制选择生成新一代种群
                # local search
                pp = np.argmax(population[i].FitnV)


        result = []
        for i in range(self.PopNum):
            result.append(self.finishing(population[i], i))  # 调用finishing完成后续工作并返回结果
        return result, rmp