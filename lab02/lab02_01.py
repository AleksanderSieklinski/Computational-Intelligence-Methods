import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
mean1 = [0, -1]
mean2 = [1, 1]
cov = [[1, 0], [0, 1]]
N_values = [2, 5, 10, 20, 100]
experiments = 10
for N in N_values:
    accuracies = []
    for i in range(experiments):
        X1 = np.random.multivariate_normal(mean1, cov, 400)
        X2 = np.random.multivariate_normal(mean2, cov, 400)
        y1 = np.ones(400)
        y2 = np.zeros(400)
        X = np.concatenate((X1, X2))
        y = np.concatenate((y1, y2))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=N/400)
        sgd_clf = SGDClassifier(loss="hinge", penalty="l2")
        sgd_clf.fit(X_train, y_train)
        y_pred = sgd_clf.predict(X_test)
        accuracies.append(accuracy_score(y_test, y_pred))
        if i == 0:
            plt.figure()
            plt.scatter(X[:, 0], X[:, 1], c=y)
            xx = np.linspace(-3, 3)
            yy = -sgd_clf.coef_[0][0]/sgd_clf.coef_[0][1] * xx - sgd_clf.intercept_/sgd_clf.coef_[0][1]
            plt.plot(xx, yy, 'k-')
            plt.title(f'N = {N}')
            plt.show()
    print(f'N = {N}, średnia dokładność: {np.mean(accuracies)}, odchylenie standardowe: {np.std(accuracies)}')