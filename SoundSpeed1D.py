'''
@Author : NONO SAHA Cyrille Merleau 
@Email : csaha@aims.edu.gh

Speed of Sound simulation using Symbolic Regression 
'''

#Necessary libraries 
import numpy as np 
import array
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import Image
import pydotplus
import scipy.stats as scs
from gplearn.genetic import SymbolicRegressor, check_random_state, SymbolicTransformer
import gplearn as gp 

#Definition of some useful functions 
def observe(x) : 
    f = np.sin(np.exp(np.cos(x**2) - 2*x) * np.cos(2*x**3))
    return f

def estimate(X0) : 
    f = 0 #Here is where you can put the generated function by gplearn
    return f
#Protected sqrt function 
def sqrt(x) : 
    return np.sqrt(abs(x))

#Protected cos function 
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
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(np.abs(x1) > 0.001, np.exp(x1), 1.)

#Main function 
def main() : 
    # Generate the value of x 
    x_test_values = np.arange(1,2.01,0.01)
    x_observed_value = array.array('d')

    for x in x_test_values : 
        x_observed_value.append(observe(x))
    #Generate some random noise 
    noise = np.random.normal(-1,1,len(x_observed_value))
    
    #Add the noise to a generated Data 
    #Y_train = x_observed_value + noise
    
    Y_train = [-2.32969664, -2.65968158 ,-2.72572594 ,-0.51148277, -1.32901313, -2.24677575
    ,1.04095231, -0.84369042 ,-0.76302401 ,-0.19910328 , 0.58804057 ,-2.77344968,
    -0.37498052 ,-0.0533432,  -2.71957971, -1.72341419, -1.76671318, -1.61307711,
    -0.80547973,  0.56839139 ,-1.00452292, -0.99525049 ,-0.45457569, -1.84295339,
    -0.57577385, -1.43824884 , 0.06017463 , 0.4112779,  -1.50514896, -0.47799747,
     0.87055506 ,-0.57908788 ,-1.27457579 ,-0.7545246  , 1.68084936  ,0.04289561,
     -2.31880285, -1.65098291 , 0.86623044 ,-1.17092527 ,-0.38352294 ,-0.22982717,
    -1.37789249, -2.12589998 ,-0.87767716, -0.29430013, -1.25344852 , 0.62044064,
     -0.94901718 ,-0.87706991 ,-0.34404281 ,-2.67282963 , 0.02482819 ,-1.09130232,
    -1.12730333 ,-1.57134418 ,-0.49390292 ,-2.33060991 ,-0.38663013 ,-1.30056801,
    -2.82323864 ,-2.84797509, -0.81053977, -1.9357591,  -0.67323035, -2.47417178,
    -1.3709051 , -2.48637129 ,-0.09552908,  1.27959327, -3.35335561, -2.31205747,
    -2.65294456 ,-1.45209591, -0.33857464 ,-1.03443114 ,-2.4123324 , -0.71717612,
     0.53066943, -1.30125161 ,-3.03606909, -0.62925414, -1.90745934 , 0.11277146,
    -3.84600099 ,-1.27699321 ,-0.47245481, -1.96211105, -0.13228734, -2.44206814,
    -0.21958032 ,-0.44864997 ,-0.03580324 ,-1.26049182, -2.02978362, -2.33455222,
    -1.35459324  ,1.32652986 , 2.0485775 , -0.38286748 ,-0.72387812]
   
    #print x_observed_value, Y_train
    
    '''
        Apply the Symbolic regression to find the function that you started with
    '''
    exp = gp.functions.make_function(function=_protected_exp,name="exp",arity=1)
    #print np.array([x_test_values])
    mygp = SymbolicRegressor(population_size=10000,
                           generations=50, stopping_criteria=0.01,
                           p_crossover=0.7, p_subtree_mutation=0.1,
                           p_hoist_mutation=0.05, p_point_mutation=0.1,
                           max_samples=1, verbose=1,
                           parsimony_coefficient=0.01, random_state=0,function_set=('add','mul','div','sub','cos','sin',exp)
                           ,const_range =(0,1))


    #fitting the model 
    print len(Y_train), len(x_test_values)
    mygp.fit(np.array([x_test_values]).T,np.array(Y_train).T)

    #print the function tree generated
    print mygp._program
    '''
    #Chi-Squart Statistic test on the generated model
    experimentData = array.array('d')
    for x in x_test_values : 
        experimentData.append(estimate(x))
    print experimentData
    #plt.plot(x_test_values,experimentData)
    #plt.plot(x_test_values,Y_train)
    #plt.show()
    print Y_train
    print "Chi-Square Value is = : ", scs.chisquare(Y_train,experimentData,axis=0) 
    '''


#Run the main function by default 
if __name__ == "__main__" : 
    main()