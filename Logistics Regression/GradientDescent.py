# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from math import exp

plt.rcParams["figure.figsize"] = (10, 6)

# Helper function to normalize data
def normalize(X):
    return X - X.mean()

# Method to make predictions
def predict(X, b0, b1):
    return np.array([1 / (1 + exp(-1*b0 + -1*b1*x)) for x in X])


# Method to train the model
def logistic_regression(X, Y):
    X = normalize(X)

    # Initializing variables
    b0 = 0
    b1 = 0
    L = 0.0001
    epochs = 300

    for epoch in range(epochs):
        y_pred = predict(X, b0, b1)
        D_b0 = sum(y_pred - Y)  # Derivative of loss wrt b0
        D_b1 = sum(X * (y_pred - Y))  # Derivative of loss wrt b1
        # Update b0 and b1
        b0 = b0 - L * D_b0
        b1 = b1 - L * D_b1

    return b0, b1

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    data = pd.read_csv("Social_Network_Ads.csv")
    data.head()



    # Visualizing the dataset
    # plt.scatter(data['Age'], data['Purchased'])
    #
    # Divide the data to training set and test set
    X_train, X_test, y_train, y_test = train_test_split(data['Age'], data['Purchased'], test_size=0.20)


    # Training the model
    b0, b1 = logistic_regression(X_train, y_train)

    # Making predictions
    X_test_norm = normalize(X_test)

    y_pred = predict(X_test_norm, b0, b1)

    y_pred = [1 if p >= 0.5 else 0 for p in y_pred]

    plt.clf()
    plt.scatter(X_test, y_test)
    plt.scatter(X_test, y_pred, c="red")
    plt.show()

    # The accuracy
    accuracy = 0
    for i in range(len(y_pred)):
        if y_pred[i] == y_test.iloc[i]:
            accuracy += 1
    print(f"Accuracy = {accuracy / len(y_pred)}")