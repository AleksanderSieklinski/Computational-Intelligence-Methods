import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
def fun(X_train, X_test, y_train, y_test, split):
    clf = Perceptron(tol=1e-3, random_state=0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Dokładność: {accuracy} dla podziału {split}')
    cm = confusion_matrix(y_test, y_pred)
    plt.imshow(cm, interpolation='nearest',cmap=plt.cm.Blues)
    plt.title('Macierz pomyłek')
    plt.colorbar()
    tick_marks = np.arange(len(iris.target_names))
    plt.xticks(tick_marks, iris.target_names)
    plt.yticks(tick_marks, iris.target_names)
    plt.tight_layout()
    plt.ylabel('Prawdziwe etykiety')
    plt.xlabel('Przewidywane etykiety')
    plt.show()
iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
fun(X_train, X_test, y_train, y_test, 0.2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
fun(X_train, X_test, y_train, y_test, 0.3)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)
fun(X_train, X_test, y_train, y_test, 0.4)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)
fun(X_train, X_test, y_train, y_test, 0.5)