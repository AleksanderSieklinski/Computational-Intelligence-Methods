import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)
accuracies = []
epochs = range(1, 100)
for epoch in epochs:
    clf = Perceptron(max_iter=epoch, tol=None, early_stopping=False, random_state=0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)
plt.plot(epochs, accuracies)
plt.xlabel('Liczba epok')
plt.ylabel('Dokładność')
plt.title('Dokładność klasyfikacji w zależności od liczby epok')
plt.show()