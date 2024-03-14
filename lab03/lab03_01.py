import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Pobranie i wczytanie pliku medicine.txt
df = pd.read_csv('medicine.txt', sep=',')

# Normalizacja danych
scaler = StandardScaler()
df[['Presence 1', 'Presence 2']] = scaler.fit_transform(df[['Presence 1', 'Presence 2']])

# Podział na zbiór treningowy i testowy
X = df[['Presence 1', 'Presence 2']]
y = df['Was medicine effective?']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)

# Definicja różnych konfiguracji sieci neuronowych
configs = [(10,), (20,), (10, 10), (20, 20), (10, 10, 10), (20, 20, 20), (10, 10, 10, 10), (20, 20, 20, 20), (20, 20, 20, 20, 20)]
for config in configs:
    # Stworzenie i trening sieci neuronowej
    mlp = MLPClassifier(hidden_layer_sizes=config, max_iter=1000, random_state=1)
    mlp.fit(X_train, y_train)

    # Predykcja na zbiorze testowym
    predictions = mlp.predict(X_test)

    # Obliczenie skuteczności
    accuracy = accuracy_score(y_test, predictions)

    print(f"Configuration: {config}, Accuracy: {accuracy * 100:.2f}%")

    # Rysowanie granic decyzyjnych
    x_min, x_max = X['Presence 1'].min() - .5, X['Presence 1'].max() + .5
    y_min, y_max = X['Presence 2'].min() - .5, X['Presence 2'].max() + .5
    h = .02
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = mlp.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X['Presence 1'], X['Presence 2'], c=y, cmap=plt.cm.Spectral)
    plt.title(f'Neural Network with configuration {config}')
    plt.show()