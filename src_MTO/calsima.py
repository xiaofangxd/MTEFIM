import geatpy as ea  # 导入geatpy库
import numpy as np
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


class calsima(MTOsogaAlgorithm):
    """
    calsima : class -  single task GA(单任务遗传算法)
模板说明:
    该模板是普通的遗传算法，可处理多个任务的版本，每个任务对应的进化求解器是普通的遗传算法，他们之间没有发生知识迁移
    注意：本算法模板中的problem和population为一个存储种群类对象的列表，而不是单个种群类对象。种群是随机初始化的


"""

    def __init__(self, problem, population, filenameall):
        if type(population) != list or type(problem) != list:
            raise RuntimeError('传入的问题或者种群对象列表必须为list类型')
        MTOsogaAlgorithm.__init__(self, problem, population)  # 先调用父类构造方法
        self.filenameall = filenameall
        self.name = 'stGA'
        self.PopNum = self.ProNum  # 种群数目
        # 为不同的种群设置不同的选择、重组、变异算子
        self.selFunc = ['tour' for _ in range(self.PopNum)]  # 锦标赛选择算子
        self.recOpers = []
        self.mutOpers = []
        Pms = []
        Pcs = []
        for i in range(self.PopNum):  # 遍历种群列表
            Pms.append(1 / self.problem[i].Dim)
            Pcs.append(0.8)
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
        population[0].initChrom(population[0].sizes)  # 初始化种群染色体矩阵
        population[0].Chrom = repOper(population[0].Field[1, 0], population[0].Chrom, self.problem[0].G)
        self.call_aimFunc(population[0], 0)  # 计算种群的目标函数值
        population[0].FitnV = ea.scaling(population[0].ObjV, population[0].CV, self.problem[0].maxormins)  # 计算适应度
        population[1].initChrom(population[1].sizes)  # 初始化种群染色体矩阵
        population[1].Chrom = population[0].Chrom
        self.call_aimFunc(population[1], 1)  # 计算种群的目标函数值
        population[1].FitnV = ea.scaling(population[1].ObjV, population[1].CV, self.problem[1].maxormins)  # 计算适应度

        result = []
        for i in range(self.PopNum):
            result.append(population[i].ObjV)  # 调用finishing完成后续工作并返回结果
        return result