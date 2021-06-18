import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("carprices.csv")
dummydf = pd.get_dummies(df['Car Model'])

newdf =pd.concat([df,dummydf],axis=1)

X = newdf.drop(["Car Model","Sell Price($)","Mercedez Benz C class"],axis=1)
y = newdf["Sell Price($)"]
x = newdf["Mileage"]
plt.scatter(x,y,color="red",marker="+")
from sklearn import linear_model
reg = linear_model.LinearRegression()
reg.fit(X,y)
print(reg.predict([[86000,7,0,1]]))
print(reg.score(X,y))

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
df["Car Model"]=LabelEncoder.fit_transform(df["Car Model"],df["Car Model"])
X = df[["Car Model","Mileage","Age(yrs)"]].values
y = df["Sell Price($)"].values
ct = ColumnTransformer([('Car Model', OneHotEncoder(), [0])], remainder = 'passthrough')
X = ct.fit_transform(X,y)
X = X[:,1:]
reg.fit(X,y)
print(reg.predict([[1,0,86000,7]]))
print(reg.score(X,y))