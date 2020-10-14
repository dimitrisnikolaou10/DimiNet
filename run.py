import pandas as pd
from Model import LinearRegression
from Metrics import plot_errors

df = pd.read_csv("data/fish.csv")
df = pd.concat((df,pd.get_dummies(df.Species)),1)
y = df["Weight"].values.reshape((df["Weight"].values.shape[0],1))
del df["Species"], df["Weight"]
X = df.values

lr = LinearRegression()
loss_curve = lr.fit(X,y)
y_pred = lr.predict(X)


plot_errors(loss_curve)
