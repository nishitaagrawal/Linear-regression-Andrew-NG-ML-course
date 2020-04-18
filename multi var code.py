import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score 
from sklearn.preprocessing import normalize
data=pd.read_csv("D:/Desktop/multi_var.txt",header=None)
data=normalize(data)
x1=(data[0])
x2=(data[1])
Y=(data[2])
th0=0
th1=0
th2=0

#cost function 

def cost(a1,a2,b,t0,t1,t2):
    m=len(b)
    for i in range(len(Y)):
        A1=0
        for j in range(len(x1)):
            s1=((t0+t1*a1[j]+t2*a2[j])-b[j])**2
            A1+=s1
    return (1/(2*m))*A1


#Gradient Descent

alpha=0.01
no_iter=1500
m=len(Y)
for i in range(no_iter):
    M1=0
    for j in range(len(x1)):
        m1=((th0+th1*x1[j]+th2*x2[j])-Y[j])*x1[j]
        M1+=m1
    M2=0
    for k in range(len(x2)):
        m2=((th0+th1*x1[k]+th2*x2[k])-Y[k])*x2[k]
        M2+=m2
    M3=0    
    for p in range(len(Y)):
        m3=((th0+th1*x1[p]+th2*x2[p])-Y[p])
        M3+=m3
    th0=th0-(alpha/m)*M3
    th1=th1-(alpha/m)*M1
    th2=th2-(alpha/m)*M2

#hypothesis
y=th0+th1*x1+th2*x2

print(r2_score(Y,y))
