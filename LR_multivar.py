import pandas as pd
from sklearn import linear_model
from word2number import w2n
import math
df = pd.read_csv("hiring.csv")
df.experience = df.experience.fillna("zero")
df.experience = df.experience.apply(w2n.word_to_num)
df['test_score(out of 10)'] = df['test_score(out of 10)'].fillna(math.floor(df['test_score(out of 10)'].mean()))
print(df)
reg = linear_model.LinearRegression()
reg.fit(df[["experience","test_score(out of 10)",'interview_score(out of 10)']],df['salary($)'])
print(reg.predict([[2,9,6]]))
print(reg.coef_)
print(reg.intercept_)
print(reg.score(df[["experience","test_score(out of 10)",'interview_score(out of 10)']],df['salary($)']))
