import pandas as pd
from Model import LinearRegression
from Metrics import plot_errors, mean_absolute_error, mean_squared_error

df = pd.read_csv("data/fish.csv")
df = pd.concat((df, pd.get_dummies(df.Species)), 1)
y = df["Weight"].values.reshape((df["Weight"].values.shape[0], 1))
del df["Species"], df["Weight"]
X = df.values

lr = LinearRegression()
loss_curve = lr.fit(X, y)
y_pred = lr.predict(X)

errors = []
for i, y in zip(y, y_pred):
    errors.append(i[0]-y[0])

# plot_errors(errors)
mean_absolute_error(errors)
mean_squared_error(errors)

