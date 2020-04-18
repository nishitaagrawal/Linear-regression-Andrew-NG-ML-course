import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as sk
data=pd.read_csv("D:/Desktop/uni_var.txt",header=None)
data.describe()
x=data[0]
y=data[1]
X_b = np.c_[np.ones((97, 1)), x]
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
theta_best
a=theta_best[0]
b=theta_best[1]
Y=a+b*x
plt.scatter(x,y)
plt.plot(x,Y,color='red')
plt.show()
print(sk.r2_score(y,Y))
