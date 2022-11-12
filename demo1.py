import numpy as np
import matplotlib.pyplot as plt


#初始化种群,popsize代表种群个数，n代表基因长度，
def start(popsize,n):
    population=[]
    for i in range(popsize):
        pop=''
        for j in range(n):

            pop=pop+str(np.random.randint(0,2))
        population.append(pop)
    return population
#计算种群中每个个体此时所代表的解的重量和价值
def computeFitness(population,weight,value):
    total_weight = []
    total_value = []
    for pop in population:
        weight_temp = 0
        value_temp = 0
        for index in range(len(pop)):
            if pop[index] == '1':
                weight_temp += int(weight[index])
                value_temp += int(value[index])
        total_weight.append(weight_temp)
        total_value.append(value_temp)
    return  total_weight,total_value

def computesingle(single,value):
    value_temp = 0
    for index in range(len(single)):
        if single[index] == '1':
            value_temp += int(value[index])
    return value_temp

#筛选符合条件的  即小于重量限制的
def select(population,weight_limit,total_weight,total_value):
    w_temp = []
    p_temp = []
    pop_temp = []
    for weight in total_weight:
        out = total_weight.index(weight)
        if weight <= weight_limit:
            w_temp.append(total_weight[out])
            p_temp.append(total_value[out])
            pop_temp.append(population[out])
    return pop_temp,w_temp,p_temp

#进行轮盘赌 每次迭代种群数量都会下降 所以使用轮盘赌来进行选择 有点不清楚
def roulettewheel(s_pop,total_value):

    p =[0]
    temp = 0

    sum_value = sum(total_value)
    for i in range(len(total_value)):

        unit = total_value[i]/sum_value
        p.append(temp+unit)
        temp += unit
    new_population = []
    i0 = 0
    while i0 < popsize:
        select_p = np.random.uniform()
        for i in range(len(s_pop)):

            if select_p > p[i] and select_p <= p[i+1]:
                new_population.append(s_pop[i])
        i0 += 1

        # if select_p < p[0]:
        #     new_population.append(s_pop[0])
        #     i += 1
        # elif p[1] <= select_p < p[2]:
        #     new_population.append(s_pop[1])
        #     i += 1
        # for index  in range(3,len(s_pop)):
        #     if p[index - 1] < select_p <= p[index]:
        #         new_population.append(s_pop[index])
        #         i += 1
        #     break
    # print(len(new_population))
    return new_population
def ga_cross(new_population,total_value,pcross):#随机交配

    new = []

    while len(new) < popsize:
        mother_index = np.random.randint(0, len(new_population))
        father_index = np.random.randint(0, len(new_population))
        threshold = np.random.randint(0, n)#单点交叉位置
        if (np.random.uniform() < pcross):#如果随机概率小于交叉概率 才往下执行
            temp11 = new_population[father_index][:threshold]#字符串的拆分
            temp12 = new_population[father_index][threshold:]
            temp21 = new_population[mother_index][threshold:]
            temp22 = new_population[mother_index][:threshold]
            child1 = temp11 + temp21
            child2 = temp12 + temp22
            # new.append(child1)
            # new.append(child2)
            pro1 = computesingle(child1, value)
            pro2 = computesingle(child2, value)
            if pro1 > total_value[mother_index] and pro1 > total_value[father_index]:
                new.append(child1)
            else:
                if total_value[mother_index] > total_value[father_index]:
                    new.append(new_population[mother_index])
                else:
                    new.append(new_population[father_index])
            if pro2 > total_value[mother_index] and pro1 > total_value[mother_index]:
                new.append(child2)
            else:
                if total_value[mother_index] > total_value[father_index]:
                    new.append(new_population[mother_index])
                else:
                    new.append(new_population[father_index])
    return new

def mutation(new,pm):
    temp =[]
    for pop in new:
        p = np.random.uniform()#随机生成一个概率 与所定义交叉概率进行比较
        if p < pm:
            point = np.random.randint(0, len(new[0]))
            pop = list(pop)

            if pop[point] == '0':
                pop[point] = '1'
            elif pop[point] == '1':
                pop[point] = '0'
            pop = ''.join(pop)
            temp.append(pop)
        else:
            temp.append(pop)
    return temp


if __name__ == "__main__":
    value = [57,94,59,83,82,91,42,84,85,18,94,18,31,27,31,42,58,57,55,97,79,10,34,100,98,45,19,77,56,25,60,22,84,89,12,46,20,85,42,94,20,65,27,34,27,91,17,56,23,89,18,11,91,79,14,99,45,73,81,96,51,96,63,40,93,87,71,54,74,15,32,57,70,62,12,71,57,97,48,33,42,25,59,91,17,63,81,49,60,90,87,25,15,20,76,76,53,59,40,59]

    weight = [9.4, 7.4, 7.7, 7.4, 2.9, 1.1, 7.3, 9.0, 8.1, 7.2, 7.5, 4.2, 4.4, 5.7, 2.0, 2.0, 9.9, 9.5, 5.2, 7.1, 6.8, 1.6, 7.9, 3.0, 1.6, 9.0, 2.1, 4.9, 7.0,  6.8, 7.7, 2.1, 8.4, 1.9, 6.5, 3.8, 2.5, 4.3, 9.9, 8.5, 8.0, 1.0, 4.4,  2.6, 2.1, 7.4, 1.5, 2.2, 8.1, 7.9, 1.5, 3.5, 2.4, 1.6, 4.3, 7.5, 2.5, 7.6, 4.8, 6.5, 1.5, 2.3, 1.0, 8.1, 8.1, 6.7, 5.8, 7.7, 4.9, 1.6, 6.5, 7.4, 1.4, 4.1, 7.4, 7.4, 1.7, 1.2, 9.5, 1.9, 7.5, 6.1, 5.9, 3.7, 7.5, 9.0, 1.7, 7.9, 1.5, 7.8, 7.6, 9.3, 9.8, 8.0, 3.3, 3.9, 9.6, 7.1, 3.9, 3.9]

    n = len(value)
    weight_limit = 285
    pm = 0.2#变异概率
    pc = 0.8#交叉概率
    popsize = 500#初始种群个数
    iters = 10000#迭代次数
    population = start(popsize, n)
    iter = 0
    best_pop = []
    best_v = []
    best_w = []
    avg = []
    while iter < iters:#小于迭代次数进行迭代
        # print(f'第{iter}代')
        # print("初始为",population)
        w, p = computeFitness(population, weight, value)#适应度值
        # print('weight:',w,'value:',p)
        # print(w)
        # print(p)
        s_pop, s_w, s_p = select(population, weight_limit, w, p)#为了防止weight超标 进行筛选

        best_index = s_p.index(max(s_p))
        best_pop.append(s_pop[best_index])
        best_v.append(s_p[best_index])
        best_w.append(s_w[best_index])
        # print(s_pop[best_index])
        # print(s_p[best_index])
        # print(s_w[best_index])
        # print(f'筛选后的种群{s_pop},长度{len(s_pop)},筛选后的weight{s_w},筛选后的value{s_p}')
        new_pop = roulettewheel(s_pop, s_p)#进行轮盘赌
        w,p1 = computeFitness(new_pop, weight, value)
        # print(f'轮盘赌选择后{new_pop},{len(new_pop)}')
        new_pop1 = ga_cross(new_pop, p1, pc)#进行交叉
        # print(f'交叉后{len(new_pop1)}')
        population = mutation(new_pop1, pm)
        # print(population)
        # print(f'第{iter}迭代结果为{max(s_p)}')
        iter += 1#指针+1
        # print(len(population1))
    best_i = best_v.index(max(best_v))

        # print(f'实验参数为:变异阈值:{pm},交叉阈值{pc},种群数量{popsize}')
print(f'在该实验参数下，总共迭代{iters}次')
print('*'*300)
print(f'最优解为{best_pop[best_i]}')
print('*'*300)
print(f'value:{best_v[best_i]},weight:{best_w[best_i]}')

