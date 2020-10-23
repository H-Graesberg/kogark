# --> Import standard Python libraries.
import numpy as np

# --> Import sklearn utility functions to create derived-class objects.
from sklearn.base import BaseEstimator, ClassifierMixin

import csv

with open('data.csv', 'r') as file:
    data_list = []
    data_reader = csv.reader(file)
    for row in data_reader:
        data_list.append(row[1:])
    data_list.pop(0)

#print(len(data_list))

with open('targets.csv', 'r') as file:
    target_list = []
    target_reader = csv.reader(file)
    for row in target_reader:
        target_list.append(row[1])
    target_list.pop(0)

#print(len(target_list))



# --> Redefine the Heavisde function.
H = lambda x: np.heaviside(x, 1).astype(np.int)

class Rosenblatt(BaseEstimator, ClassifierMixin):
    """
    Implementation of Rosenblatt's Perceptron using sklearn BaseEstimator and
    ClassifierMixin.
    """

    def __init__(self):
        # --> Weights of the model.
        self.weights = None

        # --> Bias.
        self.bias = None

    def predict(self, X):
        return H( X.dot(self.weights) + self.bias )

    def fit(self, X, y, epochs=100):
        
        """
        Implementation of the Perceptron Learning Algorithm.
        
        INPUT
        -----
        
        X : numpy 2D array. Each row corresponds to one training example.
        
        y : numpy 1D array. Label (0 or 1) of each example.
        
        OUTPUT
        ------
        
        self : The trained perceptron model.
        """

        # --> Number of features.
        n = X.shape[1]

        # --> Initialize the weights and bias.
        self.weights = np.zeros((n, ))
        self.bias = 0.0

        # --> Perceptron algorithm loop.
        for _ in range(epochs):

            # --> Current number of errors.
            errors = 0

            # --> Loop through the examples.
            for xi, y_true in zip(X, y):

                # --> Compute error.
                error = y_true - self.predict(xi)

                if error != 0:
                    # --> Update the weights and bias.
                    self.weights += error * xi
                    self.bias += error

                    # --> Current number of errors.
                    errors += 1

            # --> If no error is made, exit the outer for loop.
            if errors == 0:
                break

        return self

