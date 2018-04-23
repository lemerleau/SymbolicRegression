'''
@author : NONO SAHA Cyrille Merleau 

@email : csaha@aims.edu.gh

Implementation of Symbolic regression using Deap library in Python. 

'''
import operator
import math
import random
from pandas import read_csv
import numpy
import numpy as np

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

# Define new functions
def protectedDiv(left, right):
    
    if right==0 : 
        return 1 
    return left/right
    '''
    try:
        return left / right
    except ZeroDivisionError:
        return 1
    '''

def protectedSqrt(x) : 
    return np.sqrt(abs(x))


def main():
    #Loading the data from the csv file
    filename = "data2D.csv"
    data = read_csv(filename,sep=",",header=0)

    #Set up the functions set .
    pset = gp.PrimitiveSet("MAIN", 2)
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    #pset.addPrimitive(operator.ipow, 2)
    pset.addPrimitive(protectedDiv, 2)
    pset.addPrimitive(protectedSqrt, 1)
    pset.addEphemeralConstant("rand101", lambda: random.randint(-100,100))
    pset.renameArguments(ARG0='T')
    pset.renameArguments(ARG1='p')

    creator.create("FitnessMin", base.Fitness, weights=(-1,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)

    def evalSymbReg(individual, data):
        # Transform the tree expression in a callable function
        func = toolbox.compile(expr=individual)
        # Evaluate the mean squared error between the expression
        # and the real function : x**4 + x**3 + x**2 + x
        sqerrors = ((func(d[0],d[1]) - d[2])**2 for d in data.get_values())
        return math.fsum(sqerrors) / len(data),

    toolbox.register("evaluate", evalSymbReg, data=data)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

    random.seed(318)

    pop = toolbox.population(n=1000)
    hof = tools.HallOfFame(1)
    
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)

    pop, log = algorithms.eaSimple(pop, toolbox, 0.6, 0.1, 100, stats=mstats,
                                   halloffame=hof, verbose=True)
    # print log
    return pop, log, hof

if __name__ == "__main__":
    pop, log, hof = main()
    print hof[0]




