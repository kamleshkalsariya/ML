
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_csv("canada_per_capita_income.csv")
plt.xlabel("year")
plt.ylabel("income")
plt.scatter(df.year,df.income,color = "red",marker = "+")
year = df.drop('income',axis='columns')
income = df.income
reg = linear_model.LinearRegression()
reg.fit(year,income)
print(reg.predict([[2020]]))
print(reg.coef_)
print(reg.intercept_)
print(reg.score(year,income))

