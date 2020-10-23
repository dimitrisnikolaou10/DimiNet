import pandas as pd
from src.Model import LogisticRegression
from src.Metrics import accuracy, confusion_matrix


def main():
    print("Training has started.")

    # df = pd.read_csv("data/fish.csv")
    # df = pd.concat((df, pd.get_dummies(df.Species)), 1)
    # y = df["Weight"].values.reshape((df["Weight"].values.shape[0], 1))
    # del df["Species"], df["Weight"]

    df = pd.read_csv("../data/titanic.csv")
    del df["Unnamed: 0"]
    df.dropna(how="any", inplace=True)
    y = df["Survived"].values.reshape((df["Survived"].values.shape[0], 1))
    del df["Survived"]

    X = df.values
    # lr = LinearRegression()
    lr = LogisticRegression(regularize="RidgeClassification")
    loss_curve = lr.fit(X, y)
    y_pred = lr.predict(X)

    # errors = []
    # for i, y in zip(y, y_pred):
    #     errors.append(i[0]-y[0])

    # plot_errors(errors)
    # mean_absolute_error(errors)
    # mean_squared_error(errors)
    accuracy(y, y_pred)
    cm = confusion_matrix(y, y_pred)
    print(cm)


if __name__ == "__main__":
    main()

