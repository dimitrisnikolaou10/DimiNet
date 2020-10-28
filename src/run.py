import pandas as pd
import numpy as np
from src.Model import LogisticRegression, LinearRegression
from src.Metrics import accuracy, confusion_matrix, mean_squared_error, mean_absolute_error, plot_errors


def main():
    print("Training has started.")

    regression = True

    if regression:

        df = pd.read_csv("../lib/Fish.csv")
        df = pd.concat((df, pd.get_dummies(df.Species)), 1)
        y = df["Weight"].values.reshape((df["Weight"].values.shape[0], 1))
        del df["Species"], df["Weight"]

        X = df.values
        lr = LinearRegression()
        loss_curve = lr.fit(X, y, optimization="GradientDescent", epochs=np.power(10, 6))
        y_pred = lr.predict(X)

        errors = []
        for i, y in zip(y, y_pred):
            errors.append(i[0]-y[0])

        mean_absolute_error(errors)
        mean_squared_error(errors)
        # plot_errors(loss_curve)

    else:

        df = pd.read_csv("../lib/titanic.csv")
        del df["Unnamed: 0"]
        df.dropna(how="any", inplace=True)
        y = df["Survived"].values.reshape((df["Survived"].values.shape[0], 1))
        del df["Survived"]

        X = df.values
        lr = LogisticRegression(regularize="RidgeClassification")
        loss_curve = lr.fit(X, y, print_loss_every=1000)
        y_pred = lr.predict(X)


        accuracy(y, y_pred)
        cm = confusion_matrix(y, y_pred)
        print(cm)


if __name__ == "__main__":
    main()

