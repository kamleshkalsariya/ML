import pandas as pd
import numpy as np
from sklearn import linear_model
df = pd.read_csv("test_scores.csv")
df = df.drop(["name"],axis = 1)
x= np.array(df.math)
y = np.array(df.cs)
reg = linear_model.LinearRegression()
reg.fit(df.drop(["cs"],axis=1),df.cs)
print(reg.coef_)
print(reg.intercept_)
def gradient_descent(x,y):
    m=0
    b=0
    learning_rate = 0.0002
    n = len(x)

    for i in range(100000):
        y_pre = m*x+b
        cost = (1/n)*sum([value**2 for value in (y-y_pre)])
        if cost <=0.01:
            break
        md = -(2 / n) * sum(x * (y - y_pre))
        bd = -(2 / n) * sum(y - y_pre)
        m = m - learning_rate*md
        b = b- learning_rate*bd
        print("m={},b={},cost={}".format(m,b,cost))
gradient_descent(x,y)