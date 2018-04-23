'''
@author : NONO Saha Cyrille Merleau 
@email : csaha@aims.edu.gh

'''

#Import libraries 
import numpy as np 
from pandas import read_csv
import matplotlib.pyplot as plt 


#approximated functions 
def test1D(T) : 
    return -32559283.4192438*T**3 - 2.00972707906266*T*(35782.1749735695*T**3*(-86470.1*T - 386.751) + 197.245) - 200*T + 0.017084692682657/(T**2*(200*T + 919.783))

def approxg7(T,p) : 
    result = 2550.5703828312 + 2.21977155129427*p + 0.00677172893636732*T*p + 5.14747570590989e-7*(T**3)*((np.log(p))**2) - 5.54665735805161*T - 4.5745486144701e-12*p*(T**4)*(np.log(p))**2
    return result 

def approxg6(T,p) : 
    result = 2659.73893234743 +4.96616159155947*p +0.000746338251128017*(T**2)*np.sqrt(p) +1.68880310786392e-6*0.473097216332019**p*(T**3)-6.10315170084603*T -27.1117396759924*np.sqrt(p) -2.28064313757784e-6*p*(T**2)*np.sqrt(p)
    return result 

def approxg5(T,p) : 
    #result =abs(np.subtract(np.subtract(np.add(X0, np.add(divide(sqrt(np.multiply(X1, X0)), divide(np.add(log(X1), abs(np.subtract(np.add(np.add(np.add(divide(X0, -528.547), divide(X0, 180.916)), abs(np.add(X1, 272.261))), np.add(np.multiply(np.add(X1, 272.261), divide(920.160, X0)), np.subtract(X1, X0))), abs(np.subtract(np.add(X1, 272.261), X0))))), X0)), np.add(np.add(np.add(divide(np.multiply(X1, X0), 81.673), divide(np.multiply(log(np.subtract(np.add(X1, 272.261), X0)), divide(920.160, X0)), 81.673)), abs(np.subtract(588.852, np.add(X1, 272.261)))), np.add(np.multiply(np.add(divide(np.subtract(sqrt(np.subtract(np.subtract(X1, X0), np.multiply(X1, abs(np.multiply(np.subtract(divide(log(X1), 81.673), np.add(X1, X0)), log(X1)))))), abs(np.subtract(np.add(X1, 272.261), X0))), divide(np.add(sqrt(np.subtract(abs(np.multiply(X1, X0)), sqrt(np.subtract(abs(np.multiply(log(X1), X0)), np.add(X1, 272.261))))), np.multiply(np.add(divide(np.subtract(sqrt(np.subtract(X1, np.multiply(X1, np.multiply(np.subtract(X1, X0), log(X1))))), abs(np.subtract(np.add(X1, 272.261), X0))), divide(np.add(log(abs(np.subtract(np.add(X1, 272.261), X0))), np.add(log(X1), np.add(np.subtract(588.852, np.add(X1, 272.261)), np.add(np.multiply(np.add(divide(sqrt(np.add(np.add(X1, 272.261), np.subtract(X1, X0))), divide(-174.565, X0)), 272.261), divide(920.160, X0)), np.subtract(divide(np.subtract(sqrt(X1), X0), divide(np.add(abs(np.multiply(X1, X0)), X0), X0)), np.add(X1, X0)))))), X0)), 272.261), divide(920.160, X0))), X0)), 272.261), divide(920.160, X0)), np.subtract(X1, X0))))), np.add(X0, -333.249)), X0))
    #result = 2552.55592250839 + 2.35926132707656*p + 0.00583132078263312*T*p + 5.59297547212465e-7*(T**3)*(np.log(p)**2)-5.56000067899082*T -  2.37310483234317e-11 * p*(T**4)*np.log(p)
    #result = 2614.47115301871 + 0.0197778222258199*T*p + 1.18336066265504e-6*(T**3)*np.sqrt(p) - 5.90412621942518*T - 8.43533614504462e-9*p*(T**3)*np.sqrt(p)
    #resultdeap1 = np.subtract(np.multiply(-83, np.add(np.multiply(sqrt(np.add(np.add(np.multiply(sqrt(sqrt(np.multiply(np.add(np.subtract(np.divide(np.multiply(p, p), p), 78), np.multiply(-83, np.add(p, -36))), 46))), np.divide(p, p)), -36), T)), np.divide(p, p)), -36)), np.add(np.add(np.add(T, T), np.divide(np.multiply(T, T), np.multiply(sqrt(np.subtract(np.add(np.add(np.add(p, np.add(np.multiply(sqrt(np.multiply(p, np.divide(p, p))), np.multiply(p, p)), sqrt(np.multiply(np.subtract(T, np.divide(T, T)), np.add(np.multiply(sqrt(T), np.divide(p, p)), -36))))), sqrt(np.multiply(np.add(np.subtract(np.subtract(np.divide(np.multiply(sqrt(p), np.add(T, T)), np.subtract(p, 31)), p), 78), np.multiply(-83, np.add(sqrt(np.divide(np.divide(p, p), np.subtract(p, 31))), -36))), 46))), p), T)), 96))), np.multiply(p, np.divide(T, -50))))
    #resultdeap2 = np.subtract(np.multiply(61, np.multiply(sqrt(np.divide(-24, np.add(np.divide(sqrt(np.subtract(7, np.multiply(p, np.subtract(p, np.add(np.add(T, -8), T))))), sqrt(np.add(np.divide(p, T), np.add(np.subtract(np.divide(np.multiply(sqrt(p), np.add(T, 74)), np.multiply(p, T)), np.subtract(np.subtract(T, np.add(sqrt(60), np.add(44, 68))), 61)), np.divide(p, T))))), T))), 81)), np.add(np.subtract(np.add(np.add(np.subtract(np.add(np.divide(np.multiply(p, -50), np.subtract(65, 45)), np.subtract(np.subtract(T, sqrt(np.subtract(T, sqrt(np.multiply(np.multiply(np.subtract(-8, 61), np.subtract(p, 69)), np.subtract(20, np.divide(T, p))))))), sqrt(np.subtract(T, sqrt(np.multiply(np.multiply(np.subtract(-8, 61), np.subtract(p, 69)), np.subtract(20, -19))))))), sqrt(np.divide(np.add(np.divide(np.subtract(np.add(np.subtract(p, np.subtract(T, 61)), T), 44), -84), np.add(np.add(np.add(np.subtract(p, np.add(np.add(T, sqrt(p)), T)), np.subtract(p, 20)), np.divide(np.subtract(np.add(np.subtract(61, sqrt(np.divide(T, p))), T), 44), T)), np.subtract(p, T))), np.divide(83, np.multiply(np.add(p, T), p))))), T), T), np.multiply(sqrt(61), np.subtract(20, -19))), 20))
    #resultdeap1 = np.subtract(np.multiply(-83, np.add(np.multiply(np.sqrt(abs(np.add(np.add(np.multiply(np.sqrt(np.sqrt(abs(np.multiply(np.add(np.subtract(np.divide(np.multiply(p, p), p), 78), np.multiply(-83, np.add(p, -36))), 46)))), np.divide(p, p)), -36), T))), np.divide(p, p)), -36)), np.add(np.add(np.add(T, T), np.divide(np.multiply(T, T), np.multiply(np.sqrt(abs(np.subtract(np.add(np.add(np.add(p, np.add(np.multiply(np.sqrt(abs(np.multiply(p, np.divide(p, p)))), np.multiply(p, p)), np.sqrt(abs(np.multiply(np.subtract(T, np.divide(T, T)), np.add(np.multiply(np.sqrt(abs(T)), np.divide(p, p)), -36)))))), np.sqrt(abs(np.multiply(np.add(np.subtract(np.subtract(np.divide(np.multiply(np.sqrt(abs(p)), np.add(T, T)), np.subtract(p, 31)), p), 78), np.multiply(-83, np.add(np.sqrt(abs(np.divide(np.divide(p, p), np.subtract(p, 31)))), -36))), 46)))), p), T))), 96))), np.multiply(p, np.divide(T, -50))))
    resultdeap3 = np.add(np.add(np.subtract(np.add(np.subtract(np.add(np.add(p, np.subtract(np.add(np.subtract(np.multiply(-25, -16), divide(np.subtract(np.add(np.add(p, divide(-25, np.subtract(T, np.subtract(T, T)))), np.multiply(p, 53)), T), np.subtract(p, np.add(divide(np.subtract(p, -86), sqrt(np.subtract(np.subtract(T, p), p))), p)))), p), T)), np.subtract(np.add(np.subtract(np.multiply(-25, -16), p), p), T)), sqrt(np.add(np.add(sqrt(np.subtract(np.add(np.subtract(86, divide(np.multiply(-25, p), np.subtract(p, -16))), p), T)), p), np.subtract(-25, np.subtract(np.subtract(p, p), sqrt(p)))))), np.subtract(np.add(np.subtract(np.multiply(-25, -16), divide(divide(-46, np.add(divide(T, np.subtract(p, divide(np.multiply(-25, -16), 86))), -16)), np.subtract(-86, np.add(sqrt(45), np.subtract(np.subtract(T, 86), 86))))), p), T)), sqrt(np.add(np.add(np.subtract(86, divide(T, np.subtract(p, divide(np.multiply(-25, -16), 86)))), p), -25))), sqrt(np.subtract(np.subtract(np.subtract(np.subtract(p, p), divide(np.add(-25, T), sqrt(T))), T), 86))), np.add(np.add(p, np.add(np.add(divide(np.add(np.add(divide(np.subtract(p, -86), sqrt(np.subtract(np.subtract(T, 86), 86))), p), np.subtract(np.add(np.subtract(np.multiply(-25, -16), divide(np.subtract(np.subtract(T, -16), 86), np.subtract(p, T))), p), T)), sqrt(p)), p), np.subtract(np.add(np.subtract(np.multiply(-25, -16), divide(np.subtract(np.subtract(-25, T), 86), np.subtract(p, T))), p), T))), np.multiply(-25, -16)))
    #rest = -4*T + 6*p - (-T - 111)/(-T + p) + sqrt(abs(T - 2*p))*(-T + 54*p - 25/T)/(p + 86) + sqrt(abs(-T - 86 - (T - 25)/sqrt(T))) - sqrt(abs(-T/(p - 4) + p + 61)) - sqrt(abs(sqrt(p) + p + sqrt(abs(-T + p + 25*p/(p + 16) + 86) - 25))) + 2000 + 46/((T/(p - 4) - 16)*(-T - 3*sqrt(5) + 86)) + (-T + 2*p + 400 + (p + 86)/sqrt(abs(T - 172)) - (T - 70)/(-T + p))/sqrt(p)
    #rest = -4*T + 2*p - np.sqrt(30)*(-p - 39) + np.sqrt(abs((60*p**2 + 60*(-T + np.sqrt(30)*(p + 39) + np.sqrt(abs(2*p - 39)) + 144)/np.sqrt(p))/(np.sqrt(30)*(p + 39) - np.sqrt(abs(T + p + 5607*p/(2*p - 39))) - 63*np.sqrt(30) + 133))) + np.sqrt(abs((p**2 - 63*np.sqrt(30) - 74 + p*(T - 3)/T)*(T - T/p + p - 63*np.sqrt(29) + 4)/(-2*T - 3*p + np.sqrt(30)*(p + 33) + (p + 39)*np.sqrt(abs(2*p - 39)) + np.sqrt(30)*(p + 39) - 63*np.sqrt(30) + 141))) + np.sqrt(abs(14*T*p/(2*p - 39) + 2*T - p*(-T/p + 2*p + 44)/np.sqrt(abs(T/p - p)) - 63)) + 63*np.sqrt(30) + 1354 + 2*p*(T - 3)/T
    rest  = 6.95897107874203 - 2194.59122457971*T - 89338.80860464*T**2  
    return rest
#Mean absolute error 
def mean_absolute_error(y, y_pred):
    """Calculate the mean absolute error."""
    return np.average(np.abs(y_pred - y))


def mean_square_error(y, y_pred):
    """Calculate the mean square error."""
    return np.average(((y_pred - y) ** 2))


def root_mean_square_error(y, y_pred):
    """Calculate the root mean square error."""
    return np.sqrt(np.average(((y_pred - y) ** 2)))

def percentage_error(y, y_pred) : 
    """Calculate the percent error"""
    R = []
    for i in range(len(y)) : 
        R.append(max(y[i],y_pred[i])/(min(y[i],y_pred[i])-1))
    
    return sum(R) 

def percent_error(y, y_pred) : 
    """Calculate the percent error """
    R = []
    for i in range(len(y)) :
        r= abs((y[i]-y_pred[i])/y_pred[i])*100 
        R.append(r)
        if r == 10.493139531864433: 
            print y[i], y_pred[i]

    return max(R),min(R)
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
#Main function 
def main () : 
    #TODO 
    print "Error function plot ....."
    #Load data from a csv file 
    #names = ['T','p','S']
    filename = "data2D.csv"
    data = read_csv(filename,sep=",",header=0)

    filename = "data1d.csv"
    data1D = read_csv(filename,sep=",",header=0)

    filename = "data1Dprime.csv"
    data1Dprime = read_csv(filename,sep=",",header=0)

    values1D = []
    pred1D = []

    for d in data1Dprime.get_values() :
        values1D.append(test1D(d[0])) 
        pred1D.append(d[1])


    
    valuesg7 = []
    valuesg6 = []
    valuesg5 = []
    predict = []
    for d in data.get_values() : 
        #print d[2], approxg7(d[0],d[1])
        predict.append(d[2])
        valuesg7.append(approxg7(d[0],d[1]))
        valuesg6.append(approxg6(d[0],d[1]))
        valuesg5.append(approxg5(d[0],d[1]))

    '''
    print "================================Mean absolute error ================================================"
    print mean_absolute_error(np.array(valuesg7),np.array(predict)), mean_absolute_error(np.array(valuesg6),np.array(predict)),percent_error(np.array(valuesg5),np.array(predict))
    
    print "================================Mean square error =================================================="
    print mean_square_error(np.array(valuesg7),np.array(predict)), mean_square_error(np.array(valuesg6),np.array(predict)),percent_error(np.array(valuesg6),np.array(predict))
    
    print "================================Root Mean square error =================================================="
    print root_mean_square_error(np.array(valuesg7),np.array(predict)), root_mean_square_error(np.array(valuesg6),np.array(predict)),percent_error(np.array(valuesg7),np.array(predict))
    '''
    print "================================1D Percent error  =================================================="
    print percent_error(np.array(values1D),np.array(pred1D))

    print "================================Root error  =================================================="
    print root_mean_square_error(np.array(values1D),np.array(pred1D))
    
    #plt.plot(errorg7)
    #plt.show()

#Run the main function by default 
if __name__=="__main__" : 
    main()


