'''
@Author : NONO SAHA Cyrille Merleau 
@Email : csaha@aims.edu.gh

Speed of Sound simulation using gplearn Symbolic Regression 
'''

#Necessary libraries 
import numpy as np 
from pandas import read_csv
import array
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import Image
import pydotplus
import scipy.stats as scs
from gplearn.genetic import SymbolicRegressor, check_random_state, SymbolicTransformer
import gplearn as gp 
import csv

#Definition of some useful functions 
def observe(x) : 
    f = np.sin(np.exp(np.cos(x**2) - 2*x) * np.cos(2*x**3))
    return f
def test(T) : 
    return 2.60499980793583e-5*1.02507519614662**T + 7.10923303645526e-8*T**3 + 2.74208894269624*3.13129661675052e-5**((3.14824875670119e-6*T**2)**2.14118323976238) - 2.71128310313156 - 8.04700679962298e-6*(T**2)*3.13129661675052e-5**((3.14824875670119e-6*T**2)**2.14118323976238)
    #return 0.0146420221969004*T + 5.04267532525505e-13*T**5 + 0.26721550731968*T*0.00024121781383422**(0.00504706639538683*T) + 24859583.2183608*0.00024121781383422**(0.00504706639538683*T)*(0.00227622476560452*T)**90.4330911353217 - 1.29028478814847 - 4.52190829804809e-5*T**2 
def estimate(X0) : 
    f =np.multiply(np.add(divide(np.subtract(np.add(X0,X0),np.exp(X0)),np.multiply(np.multiply(0.549,X0),divide(-3.033,3.779))),np.multiply(np.cos(np.add(X0,-4.093)),np.cos(np.exp(-0.766)))),np.cos(np.add(divide(np.multiply(3.760,X0),np.subtract(5.762,3.742)),np.cos(np.cos(-4.345)))))#Here is where you can put the generated function by gplearn
    return f
#Protected sqrt function 
def sqrt(x) : 
    return np.sqrt(abs(x))

#Protected cos function gm
def cos(x) : 
    return np.cos(x)

#Protected divide function 
def divide(a,b) : 
    if b==0 : 
        return 1 
    return np.divide(a,b)

#Protected log function 
def log(x) : 
    return np.log(abs(x))

#Customized gplearn function 
def _protected_exp(x1):
        with np.errstate(over='ignore'):
            a = np.where(np.abs(x1) < 100, np.exp(x1), 0.)
            #print a 
            return a


#Main function 
def main() : 
    # Generate the value of x 
    #x_test_values = np.arange(1,2.01,0.01)
    x_test_values =[]
    x_observed_value = array.array('d')

    for x in x_test_values : 
        x_observed_value.append(observe(x))
    #Generate some random noise 
    noise = np.random.normal(-0.001,0.001,len(x_observed_value))
    
    #Add the noise to a generated Data 
    #Y_train = x_observed_value + noise
    
    #print x_observed_value, Y_train
    '''
    #Save my data in a csv file 
    with open('data1Dprime.csv', 'w') as csvfile:
        fieldnames = ['T', 'P']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for d in Y:
            writer.writerow({'T': d[0], 'P': d[1]})
    '''
    '''
        Apply the Symbolic regression to find the function that you started with
    '''
    exp = gp.functions.make_function(function=_protected_exp,name="exp",arity=1)
    #print np.array([x_test_values])
    mygp = SymbolicRegressor(population_size=10000, init_depth=(2,6),tournament_size=100,
                           generations=100, stopping_criteria=0.000001,
                           p_crossover=0.5, p_subtree_mutation=0.3,
                           p_hoist_mutation=0.05, p_point_mutation=0.1,
                           max_samples=0.9, verbose=1,
                           parsimony_coefficient=0.01, random_state=0,function_set=('sub','mul','add')
                           ,const_range =(-1000,1000))

    #Load data from csv file using pandas library 
    filename = "data1Dprime.csv"
    data = read_csv(filename,sep=",",header=0)

    Y_train = []
    
    for d in data.get_values() : 
        Y_train.append(d[1])
        x_test_values.append(d[0]*100)

    
    #fitting the model 
    print len(Y_train), len(x_test_values)
    mygp.fit(np.array([x_test_values]).T,np.array(Y_train).T)

    #print the function tree generated
    print mygp._program
    '''
    #Chi-Squart Statistic test on the generated model
    p = np.arange(150,430,1)
    experimentData = array.array('d')
    print p
    for x in p : 
        experimentData.append(test(x))
    #print experimentData
    filename = "data1d.csv"
    data = read_csv(filename,sep=",",header=0)
    Pexact = []
    Yexact = []
    for d in data.get_values() : 
        Pexact.append(d[0])
        Yexact.append(d[1])
    
    fig, ax = plt.subplots()
    ax.plot(Pexact,Yexact,"r*",label="Initial")
    ax.plot(p,experimentData,label="Approximation")
    ax.legend(loc='upper left', shadow=True, fontsize='12')
    plt.title(r"1D GP Approximation :  $P=f(T)$")
    plt.xlabel(r'$T$')
    plt.ylabel(r'$P$')
    plt.savefig('Test1D2.pdf')
    #plt.plot(x_test_values,Y_train)
    plt.show()
    #print Y_train
    #print "Chi-Square Value is = : ", scs.chisquare(Y_train,experimentData,axis=0) 
    
   '''

#Run the main function by default 
if __name__ == "__main__" : 
    main()