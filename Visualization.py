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
import csv

#Definition of some useful functions 
def estimate(T) : 
    #f =np.subtract(np.subtract(np.subtract(np.subtract(np.multiply(np.sin(np.multiply(np.exp(np.add(-0.352, X0)), np.add(np.subtract(X0, 0.179), np.exp(0.281)))), np.exp(np.multiply(np.exp(np.sin(np.add(X0, np.sin(X0)))), np.sin(np.multiply(X0, X0))))), np.sin(np.sin(X0))), np.add(np.cos(X0), np.sin(X0))), np.cos(np.subtract(np.exp(X0), np.exp(np.sin(X0))))), np.cos(np.multiply(X0, X0)))
    #f =np.multiply(np.exp(np.exp(np.cos(np.subtract(np.cos(np.cos(np.multiply(np.exp(X0), np.sin(-0.625)))), np.multiply(X0, X0))))), np.sin(np.subtract(np.cos(np.sin(np.multiply(X0, X0))), np.multiply(np.subtract(-0.195, X0), np.exp(X0)))))
    #f = np.multiply(np.exp(np.exp(np.cos(np.subtract(np.cos(np.multiply(np.exp(0.477), np.subtract(X0, 0.963))), np.multiply(X0, X0))))), np.sin(np.subtract(np.cos(np.sin(np.multiply(X0, X0))), np.multiply(np.subtract(-0.190, X0), np.exp(X0)))))
    #f = np.multiply(np.exp(np.multiply(np.subtract(X0, -0.781), np.multiply(np.cos(np.subtract(np.cos(X0), np.multiply(X0, 0.794))), np.exp(np.cos(X0))))), np.sin(np.subtract(np.multiply(X0, 0.480), np.multiply(np.subtract(-0.173, X0), np.exp(X0)))))
    #f= 0.167945022209528*x + 0.0823971100798616*x*np.cos(17.4766840999811/x) + 1.92168434425551e-17*(5.52302461573631*x - np.cos(17.4766840999811/x))**16.333286849089/x + 1.13289221637823e-18*(5.52302461573631*x - np.cos(17.4766840999811/x))**16.333286849089*np.cos(17.4766840999811/x) - 0.27831899331308 - 0.0437500397538583*np.cos(17.4766840999811/x) - 8.65308344591467e-18*(5.52302461573631*x - np.cos(17.4766840999811/x))**16.333286849089
    #f=-2876.132448*x + 8.14413387159786
    #f= 8.1603154015448 - 2881.13073910938*x
    #fgp1= -120940.0*x**2 - 1926.8*x + 6.441
    #fgp2= -295571.11274871*T**2 - 100*T + 0.00752178843341362/T
    #fEq2 = 6.95897107874203 - 2194.59122457971*T - 89338.80860464*T**2  
    fgp3 = 8.1603154015448 - 2881.13073910938*T  
    return fgp3
#Protected sqrt function 
def sqrt(x) : 
    return np.sqrt(abs(x))


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
    #Load data from csv file using pandas library 
    filename = "data1Dprime.csv"
    data = read_csv(filename,sep=",",header=0)

    Y_train = []
    

    
    #Chi-Squart Statistic test on the generated model
    #p = np.arange(150,430,1)
    
    #print experimentData
    filename = "data1Dprime.csv"
    data = read_csv(filename,sep=",",header=0)
    Pexact = []
    Yexact = []
    for d in data.get_values() : 
        Pexact.append(d[0])
        Yexact.append(d[1])
    
    experimentData = array.array('d')
    for x in  Pexact : 
        experimentData.append(estimate(x))
    fig, ax = plt.subplots()
    ax.plot(Pexact,Yexact,"r",label="Initial")
    ax.plot( Pexact,experimentData,label="Approximation")
    ax.legend(loc='upper right', shadow=True, fontsize='12')
    plt.title(r"1D Eureqa GP Approximation $P_3$ :  $P= f(T)$")
    plt.xlabel(r'$T$')
    plt.ylabel(r'$P$')
    plt.savefig('Test1DEqP4.pdf')
    plt.show()
    


#Run the main function by default 
if __name__ == "__main__" : 
    main()