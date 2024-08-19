import math

import random  # 确保导入了random模块
from random import randrange
from random import choices
import numpy as np
geneNum = 100  # 种群数量
generationNum = 500  # 迭代次数


CENTER = 20 # 配送中心

# HUGE = 9999999
# PC = 1   #交叉率,没有定义交叉率，也就是说全部都要交叉，也就是1
PM = 0.1  # 变异率   以前是用的vary

n = 33 # 客户点数量
m = 2  # 换电站数量
k = 2  # 车辆数量
Q = 10 # 额定载重量, t

length = n+m+1

# 坐标   第0个换电站   1-14是顾客   16是图书馆 17到31是顾客       行驶距离要通过这个坐标自己来算
dis_net = [[0, 0.7, 1.4, 1.9, 3.0, 3.5, 0.6, 1.2, 1.8, 2.5, 3.1, 3.9, 0.9, 2.4, 2.9, 2.9, 3.2, 4.0, 2.0, 3.7, 4.4, 4.8, 5.3, 5.4, 2.5, 4.5, 5.1, 5.3, 5.9, 6.0, 3.3, 4.0, 4.6, 5.2, 5.6, 5.9],
[0.7, 0, 0.7, 1.2, 2.3, 2.8, 1.1, 0.5, 1.1, 1.8, 2.4, 3.2, 1.6, 1.7, 2.2, 2.2, 2.5, 3.3, 2.7, 3.5, 4.2, 4.6, 4.6, 4.7, 3.2, 4.3, 4.9, 5.1, 5.4, 5.3, 4.0, 4.7, 5.3, 5.7, 6.1, 5.8],
[1.4, 0.7, 0, 0.5, 1.6, 2.1, 1.8, 1.2, 0.4, 1.1, 1.7, 2.5, 2.1, 2.4, 1.5, 1.5, 1.8, 2.6, 3.2, 4.2, 3.7, 4.1, 3.9, 4.0, 3.7, 5.0, 4.4, 4.6, 4.7, 4.6, 4.5, 5.2, 5.4, 5.2, 5.4, 5.1],
[1.9, 1.2, 0.5, 0, 1.1, 1.6, 2.3, 1.7, 0.9, 0.9, 1.4, 2.0, 2.6, 2.9, 2.0, 1.3, 1.6, 2.4, 3.7, 4.7, 4.2, 4.5, 3.7, 3.8, 4.2, 5.5, 4.9, 5.1, 4.5, 4.4, 5.0, 5.7, 5.9, 5.7, 5.2, 4.9],
[3.0, 2.3, 1.6, 1.1, 0, 0.5, 3.4, 2.8, 2.0, 0.9, 0.3, 0.9, 3.7, 4.0, 3.1, 1.7, 1.4, 1.8, 4.8, 5.8, 5.3, 4.3, 3.5, 3.2, 5.3, 6.6, 6.0, 4.9, 4.3, 3.8, 6.1, 6.8, 7.0, 5.8, 4.6, 4.3],
[3.5, 2.8, 2.1, 1.6, 0.5, 0, 3.9, 3.3, 2.1, 1.4, 0.8, 0.4, 4.2, 4.5, 2.8, 1.8, 1.9, 1.3, 5.3, 6.3, 5.2, 4.8, 3.4, 2.7, 5.8, 7.1, 5.8, 5.3, 3.8, 3.3, 6.6, 7.3, 6.5, 5.5, 4.1, 3.8],
[0.6, 1.1, 2.0, 2.5, 3.3, 4.1, 0, 0.6, 1.7, 2.4, 3.0, 3.9, 0.3, 1.8, 2.8, 2.8, 3.1, 3.9, 1.4, 3.1, 3.8, 4.2, 5.0, 5.3, 1.9, 3.9, 4.5, 4.7, 5.3, 5.8, 2.7, 3.4, 4.0, 4.6, 5.0, 5.3],
[1.2, 0.5, 1.2, 1.7, 2.7, 3.3, 0.6, 0, 1.1, 1.8, 2.4, 3.3, 0.9, 1.2, 2.2, 2.2, 2.5, 3.3, 2.0, 3.0, 3.7, 4.1, 4.6, 4.7, 2.5, 3.8, 4.4, 4.6, 5.2, 5.3, 3.3, 4.0, 4.6, 5.2, 5.6, 5.8],
[1.8, 1.1, 0.4, 0.9, 1.6, 2.5, 1.7, 1.1, 0, 0.7, 1.3, 2.2, 2.0, 2.3, 1.1, 1.1, 1.4, 2.2, 3.1, 4.0, 3.3, 3.7, 3.5, 3.6, 3.6, 4.9, 4.0, 4.2, 4.3, 4.2, 4.4, 5.1, 5.0, 4.8, 5.0, 4.7],
[2.5, 1.8, 1.1, 0.9, 0.9, 1.9, 2.4, 1.8, 0.7, 0, 0.6, 1.5, 2.7, 3.0, 1.4, 0.4, 0.7, 1.5, 3.8, 4.7, 4.0, 3.6, 2.8, 2.9, 4.3, 5.5, 4.7, 4.2, 3.6, 3.5, 5.1, 5.8, 5.7, 5.1, 4.3, 4.0],
[3.1, 2.4, 1.7, 1.4, 0.3, 0.8, 3.0, 2.4, 1.3, 0.6, 0, 0.9, 3.3, 3.3, 2.0, 1.0, 1.1, 1.8, 4.4, 4.9, 4.2, 4.0, 3.2, 3.2, 4.9, 5.9, 4.9, 4.6, 4.0, 3.8, 5.7, 6.4, 5.9, 5.3, 4.6, 4.3],
[3.9, 3.2, 2.5, 2.0, 0.9, 0.4, 3.9, 3.3, 2.2, 1.5, 0.9, 0, 4.2, 4.2, 2.9, 1.9, 1.7, 0.9, 5.3, 5.8, 5.1, 4.6, 3.0, 2.3, 5.8, 6.6, 5.8, 4.8, 3.4, 2.9, 6.6, 7.3, 6.5, 4.9, 3.7, 3.4],
[0.9, 1.4, 2.1, 2.6, 3.6, 4.1, 0.3, 0.9, 2.0, 2.7, 3.3, 4.2, 0, 1.5, 2.8, 3.1, 3.4, 4.2, 1.1, 2.8, 3.5, 3.9, 4.7, 5.4, 1.6, 3.6, 4.2, 4.4, 5.0, 5.5, 2.4, 3.1, 3.7, 4.3, 4.7, 5.0],
[2.4, 1.7, 2.4, 2.9, 3.9, 4.4, 1.8, 1.2, 2.3, 2.7, 3.6, 4.3, 1.5, 0, 1.3, 2.3, 2.6, 3.4, 2.6, 1.8, 2.5, 2.9, 3.7, 4.4, 3.1, 2.6, 3.2, 3.4, 4.0, 4.5, 3.9, 3.6, 4.2, 4.0, 4.4, 4.7],
[2.9, 2.2, 1.5, 2.0, 2.7, 3.2, 2.8, 2.2, 1.1, 1.4, 2.4, 3.0, 2.8, 1.3, 0, 1.0, 1.3, 2.1, 3.9, 2.9, 2.2, 2.6, 3.4, 3.5, 4.4, 3.9, 2.9, 3.1, 3.7, 4.1, 5.2, 4.5, 3.9, 3.7, 4.1, 4.4],
[2.9, 2.2, 1.5, 1.3, 1.3, 1.8, 2.8, 2.2, 1.1, 0.4, 1.0, 1.9, 3.1, 2.3, 1.0, 0, 0.3, 1.1, 4.2, 3.9, 3.2, 3.2, 2.4, 2.5, 4.7, 4.7, 3.9, 3.8, 3.2, 3.1, 5.5, 5.5, 4.9, 4.7, 3.9, 3.6],
[3.2, 2.5, 1.8, 1.6, 1.4, 1.9, 3.1, 2.5, 1.4, 0.7, 1.1, 1.7, 3.4, 2.6, 1.3, 0.3, 0, 0.8, 4.5, 4.2, 3.5, 2.9, 2.1, 2.2, 5.0, 5.0, 4.2, 3.5, 2.9, 2.8, 5.8, 5.8, 5.2, 4.4, 3.6, 3.3],
[4.0, 3.3, 2.6, 2.4, 1.8, 1.3, 3.9, 3.3, 2.2, 1.5, 1.8, 0.9, 4.2, 3.4, 2.1, 1.1, 0.8, 0, 5.3, 5.0, 4.1, 3.7, 2.1, 1.4, 5.8, 5.8, 4.7, 4.2, 2.5, 2.0, 6.6, 6.6, 5.4, 4.8, 2.8, 2.5],
[2.0, 2.5, 3.2, 3.7, 4.7, 5.2, 1.4, 2.0, 3.1, 3.8, 4.4, 5.3, 1.1, 2.6, 3.9, 4.2, 4.5, 5.3, 0, 1.7, 2.4, 2.8, 3.6, 4.3, 0.5, 2.5, 3.1, 3.3, 3.9, 4.4, 1.3, 2.0, 2.6, 3.2, 3.6, 3.9],
[3.7, 3.5, 4.2, 4.7, 5.4, 5.9, 3.1, 3.0, 4.1, 4.5, 5.1, 6.0, 2.8, 1.8, 2.9, 4.1, 4.0, 4.0, 1.7, 0, 0.7, 1.1, 1.9, 2.6, 2.2, 0.8, 1.4, 1.6, 2.2, 2.7, 2.5, 1.8, 2.4, 2.2, 2.6, 2.9],
[4.4, 4.2, 3.7, 4.2, 4.5, 5.0, 3.8, 3.7, 3.3, 3.6, 4.2, 5.1, 3.5, 2.5, 2.2, 3.2, 3.3, 3.3, 2.4, 0.7, 0, 0.4, 1.2, 1.9, 2.9, 1.5, 0.7, 0.9, 1.5, 2.0, 3.2, 2.3, 1.7, 1.5, 1.9, 2.2],
[4.8, 4.6, 4.1, 4.6, 4.9, 5.4, 4.2, 4.1, 3.7, 4.0, 4.6, 5.5, 3.9, 2.9, 2.6, 3.6, 2.9, 2.9, 2.8, 1.1, 0.4, 0, 0.8, 1.5, 3.3, 1.9, 1.0, 0.5, 1.1, 1.6, 3.4, 2.7, 1.7, 1.1, 1.5, 1.8],
[5.3, 4.6, 3.9, 3.7, 3.5, 4.0, 5.0, 4.6, 3.5, 2.8, 3.2, 3.8, 4.7, 3.7, 3.4, 2.4, 2.1, 2.1, 3.6, 1.9, 1.2, 0.8, 0, 0.7, 4.1, 2.7, 1.8, 1.3, 0.8, 1.3, 4.2, 3.1, 2.5, 1.9, 1.9, 1.8],
[5.4, 4.7, 4.0, 3.8, 3.2, 2.7, 5.3, 4.7, 3.6, 2.9, 3.2, 2.3, 5.4, 4.4, 3.5, 2.5, 2.2, 1.4, 4.3, 2.6, 1.9, 1.5, 0.7, 0, 4.8, 3.4, 2.5, 2.0, 1.1, 0.6, 4.5, 3.8, 3.2, 2.6, 1.4, 1.1],
[2.5, 3.0, 3.7, 4.2, 5.2, 5.7, 1.9, 2.5, 3.6, 4.3, 4.9, 5.8, 1.6, 3.1, 4.4, 4.7, 5.0, 5.8, 0.5, 2.2, 2.9, 3.3, 4.1, 4.8, 0, 2.0, 3.0, 3.3, 4.1, 3.9, 0.8, 1.5, 2.1, 2.7, 3.1, 3.4],
[4.5, 4.3, 5.0, 5.5, 6.0, 6.1, 3.9, 3.8, 4.8, 5.1, 5.7, 5.7, 3.6, 2.6, 3.7, 4.7, 4.8, 4.8, 2.5, 0.8, 1.5, 1.9, 2.7, 3.2, 2.0, 0, 1.0, 1.5, 2.1, 2.6, 1.7, 1.0, 1.6, 2.1, 2.5, 2.8],
[5.1, 4.9, 4.4, 4.9, 5.2, 5.3, 4.5, 4.4, 4.0, 4.3, 4.9, 4.9, 4.2, 3.2, 2.9, 3.9, 4.0, 4.0, 3.1, 1.4, 0.7, 1.0, 1.9, 2.2, 3.0, 1.0, 0, 0.5, 1.1, 1.6, 2.7, 1.6, 1.0, 1.1, 1.5, 1.8],
[5.3, 5.1, 4.6, 5.0, 4.8, 4.7, 4.7, 4.6, 4.2, 4.1, 4.5, 4.3, 4.4, 3.4, 3.1, 3.7, 3.4, 3.4, 3.3, 1.6, 0.9, 0.5, 1.3, 1.7, 3.5, 1.5, 0.5, 0, 0.6, 1.1, 2.8, 2.1, 1.2, 0.6, 1.0, 1.3],
[5.9, 5.4, 4.7, 4.5, 4.3, 4.2, 5.3, 5.2, 4.3, 3.6, 4.0, 3.8, 5.0, 4.0, 3.7, 3.2, 2.9, 2.9, 3.9, 2.2, 1.5, 1.1, 0.8, 1.1, 4.1, 2.1, 1.1, 0.6, 0, 0.5, 3.4, 2.4, 1.8, 1.2, 1.1, 1.0],
[6.0, 5.3, 4.6, 4.4, 3.8, 3.3, 5.8, 5.3, 4.2, 3.5, 3.8, 2.9, 5.5, 4.5, 4.1, 3.1, 2.8, 2.0, 4.4, 2.7, 2.0, 1.6, 1.3, 0.6, 4.6, 2.6, 1.6, 1.1, 0.5, 0, 3.6, 2.9, 2.3, 1.7, 0.8, 0.5],
[3.3, 3.8, 4.5, 5.0, 6.0, 6.5, 2.7, 3.3, 4.4, 5.1, 5.7, 6.6, 2.4, 3.9, 5.2, 5.5, 5.8, 6.6, 1.3, 3.0, 3.7, 4.1, 4.9, 5.3, 0.8, 1.7, 2.3, 2.5, 3.4, 3.1, 0, 0.7, 1.3, 1.9, 2.3, 2.6],
[4.0, 4.5, 5.2, 5.7, 6.7, 6.9, 3.4, 4.0, 5.1, 5.8, 6.4, 6.5, 3.1, 3.6, 4.7, 5.7, 5.8, 5.6, 2.0, 1.8, 2.5, 2.9, 3.7, 4.2, 1.5, 1.0, 1.6, 1.8, 2.7, 2.4, 0.7, 0, 0.6, 1.2, 1.6, 1.9],
[4.6, 5.1, 5.4, 5.9, 6.2, 5.9, 4.0, 4.6, 5.0, 5.3, 5.9, 5.5, 3.7, 4.2, 3.9, 4.9, 4.9, 4.6, 2.6, 2.4, 1.7, 2.0, 2.8, 3.2, 2.1, 1.6, 1.0, 1.2, 2.1, 1.8, 1.3, 0.6, 0, 0.6, 1.0, 1.3],
[5.2, 5.7, 5.2, 5.6, 5.4, 5.0, 4.6, 5.2, 4.8, 4.7, 5.1, 4.6, 4.3, 4.0, 3.7, 4.3, 4.0, 3.7, 3.2, 2.2, 1.5, 1.1, 1.9, 2.3, 2.7, 2.1, 1.1, 0.6, 1.2, 1.2, 1.9, 1.2, 0.6, 0, 0.4, 0.7],
[5.6, 6.1, 5.6, 5.6, 5.4, 4.9, 5.0, 5.6, 5.2, 4.7, 5.1, 4.5, 4.7, 4.4, 4.1, 4.3, 4.0, 3.6, 3.6, 2.6, 1.9, 1.5, 1.9, 2.2, 3.1, 2.5, 1.5, 1.0, 1.1, 0.8, 2.3, 1.6, 1.0, 0.4, 0, 0.3],
[5.9, 5.8, 5.1, 4.9, 4.3, 3.8, 5.3, 5.8, 4.7, 4.0, 4.3, 3.4, 5.0, 4.7, 4.4, 3.6, 3.3, 2.5, 3.9, 2.9, 2.2, 1.8, 1.8, 1.1, 3.4, 2.8, 1.8, 1.3, 1.0, 0.5, 2.6, 1.9, 1.3, 0.7, 0.3, 0]]

# 需求量
t = [0, 6, 4, 7, 5, 3, 3, 5, 3, 4, 3, 4, 4, 2, 0, 2, 8, 2, 2, 4, 0, 0, 3, 1, 5, 3, 6, 5, 6, 5, 6, 6, 2, 6, 2, 0]
def getGene(length):   
    ##先产生一个无序的列表
    data = list(range(1,length))  ##先产生一个有序的列表
    np.random.shuffle(data)   ##有序列表打乱成无序列表
    data.insert(0, CENTER)    ##在开始插入0
    data.append(CENTER)       ##在结尾插入0
    
    #再插入车
    sum = 0
    newData = []
    for index, pos in enumerate(data):
        sum += t[pos]
        if sum > Q:
            newData.append(CENTER)
            sum = t[pos]
        newData.append(pos)

    return newData
def getpop(length,geneNum):
    pop = []
    for i in range(geneNum):
        gene = getGene(length)
       
        pop.append(gene)
    return pop
##计算一个个体的适应度值
def getfit(gene):
    distCost = 0
    dist = []  # from this to next
    # 计算距离
    i = 1
    while i < len(gene):
        
        dist.append(dis_net[gene[i]][gene[i - 1]])
        i += 1
    # 距离成本
    distCost = sum(dist)     #总行驶距离
    fit = 1/distCost   ##fitness越小表示越优秀，被选中的概率越大，
    return fit
##得到整个种群的适应度列表
def getfitness(pop):
    fitness = []
    for gene in pop:
        fit = getfit(gene)
        fitness.append(fit)
    return np.array(fitness)
def select(pop,fitness):
    fitness = fitness / fitness.sum()  # 归一化
    idx = np.array(list(range(geneNum)))
    pop_idx = np.random.choice(idx, size=geneNum, p=fitness)  # 根据概率选择
    for i in range(geneNum):
        pop[i] = pop[pop_idx[i]]
    return pop
#选择路径
def moveRandSubPathLeft(gene):
   
    path = randrange(k)  # 选择路径索引，随机分成k段
    print('path:',path)
    try:
        index = gene.index(CENTER, path+1) #移动到所选索引
        # move first CENTER
        locToInsert = 0
        gene.insert(locToInsert, gene.pop(index))
        index += 1
        locToInsert += 1
        # move data after CENTER
        print('index:',index)
        try:
            print('len(gene):',len(gene))
            while gene[index] != CENTER:
                gene.insert(locToInsert, gene.pop(index))
                index += 1
                print('执行完index+1,index=',index)
                locToInsert += 1
            return gene
            # assert(length+k == len(gene))
        except:
            print('出错啦，index:',index)
            return gene
    except:
        print('0 is not in list',gene)
        return gene
# 选择复制，选择适应度最高的前 1/3，进行后面的交叉
def choose1(pop):
    num = int(geneNum/6) * 2    # 选择偶数个，方便下一步交叉
    # sort genes with respect to chooseProb
    key = lambda gene: getfit(gene)
    pop.sort(reverse=True, key=key)      ##那就是说按照适应度函数降序排序,选了适应度值最高的那1/3
    # return shuffled top 1/3   
    return pop[0:num]
##交叉一对
def crossPair(i,gene1, gene2, crossedGenes):
    gene1 = moveRandSubPathLeft(gene1)
    gene2 = moveRandSubPathLeft(gene2)
    newGene1 = []
    newGene2 = []
    # copy first paths
    centers = 0
    firstPos1 = 1
    for pos in gene1:
        firstPos1 += 1
        centers += (pos == CENTER)
        newGene1.append(pos)
        if centers >= 2:
            break
    centers = 0
    firstPos2 = 1
    for pos in gene2:
        firstPos2 += 1
        centers += (pos == CENTER)
        newGene2.append(pos)
        if centers >= 2:
            break
    # copy data not exits in father gene
    for pos in gene2:
        if pos not in newGene1:
            newGene1.append(pos)
    for pos in gene1:
        if pos not in newGene2:
            newGene2.append(pos)
    # add center at end
    newGene1.append(CENTER)
    newGene2.append(CENTER)
    # 计算适应度最高的
    key1 = lambda gene1: getfit(gene1)
    possible1 = []
    try:
        while gene1[firstPos1] != CENTER:
            newGene = newGene1.copy()
            newGene.insert(firstPos1, CENTER)
            possible1.append(newGene)
            firstPos1 += 1
        print('第{}位置:{}'.format(i,len(possible1)))
        if len(possible1) == 0:
            crossedGenes.append(newGene1)
        else:
            possible1.sort(reverse=True, key=key1)
            crossedGenes.append(possible1[0])
    except:
        print('交叉出错啦：firstPos1', firstPos1)

    key2 = lambda gene2: getfit(gene2)
    possible2 = []
    try:
        while gene2[firstPos2] != CENTER:
            newGene = newGene2.copy()
            newGene.insert(firstPos2, CENTER)
            possible2.append(newGene)
            firstPos2 += 1
        print('第{}:{}'.format(i,len(possible2)))
        if len(possible2) == 0:
            crossedGenes.append(newGene2)
        else:
            possible2.sort(reverse=True, key=key2)
            crossedGenes.append(possible2[0])
        print('交叉完成第：', i)
    except:
        print('交叉出错啦：',i)

# 交叉
def cross1(genes):
    crossedGenes = []
    for i in range(0, len(genes), 2):
        # print('gene[i]:',genes[i])
        # print('gene[i+1]:', genes[i])
        crossPair(i,genes[i], genes[i+1], crossedGenes)
        print('交叉完成')
    return crossedGenes

# 合并
def mergeGenes(genes, crossedGenes):
    # sort genes with respect to chooseProb
    key = lambda gene: getfit(gene)
    genes.sort(reverse=True, key=key)    ##先把原来的种群100按照适应度降序排列，然后，将交叉得到的32个个体替换到种群的最后32个
    pos = geneNum - 1
    for gene in crossedGenes:
        genes[pos] = gene
        pos -= 1
    return  genes


def varyOne(gene):
    varyNum = 10    
    variedGenes = []
    for i in range(varyNum):       # 先按照这种方法变异10个，选择适应度最高的那个作为变异完的子代
        p1, p2 = choices(list(range(1,len(gene)-2)), k=2)
        newGene = gene.copy()
        newGene[p1], newGene[p2] = newGene[p2], newGene[p1] # 交换
        variedGenes.append(newGene)
    key = lambda gene: getfit(gene)
    variedGenes.sort(reverse=True, key=key)
    return variedGenes[0]

# 变异
def vary(genes):
    for index, gene in enumerate(genes):
        # 精英主义，保留前三十，这个意思就是前三十个一定不变异，到后面的个体才按照变异概率来变异
        if index < 30:
            continue
        if np.random.rand() < PM:
            genes[index] = varyOne(gene)
    return genes


from tqdm import *  # 进度条
import matplotlib.pyplot as plt
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

best_fitness = []
min_cost = []
J = []
pop = getpop(length, geneNum)  # 初始种群
# 迭代
for j in tqdm(range(generationNum)):
    print('j=',j)
    chosen_pop = choose1(pop)   # 选择   选择适应度值最高的前三分之一，也就是32个种群，进行下一步的交叉
    crossed_pop = cross1(chosen_pop)   # 交叉
    pop = mergeGenes(pop, crossed_pop) # 复制交叉至子代种群
    pop = vary(pop) # under construction
    key = lambda gene: getfit(gene)
    pop.sort(reverse=True, key=key)  # 以fit对种群排序
    cost = 1/getfit(pop[0])
    print(cost)
    min_cost.append(cost)
    J.append(j)
print(J)
print(min_cost)


# key = lambda gene: getfit(gene)
# pop.sort(reverse=True, key=key)   # 以fit对种群排序
print('\r\n')
print('data:', pop[0])
print('fit:', 1/getfit(pop[0]))
t = dis_net[0][20] / 10 * 2 + dis_net[35][20] / 8 * 2 +  1/getfit(pop[0]) / 9
print(t)
plt.plot(J,min_cost, color='r')
plt.show()
