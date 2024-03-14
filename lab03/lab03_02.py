from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Pobranie zbioru danych
digits = datasets.load_digits()

# Podział na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=42)

# Stworzenie wielowarstwowej sieci neuronowej
mlp = MLPClassifier(hidden_layer_sizes=(100,50), max_iter=1000, random_state=1)

# Trening sieci
mlp.fit(X_train, y_train)

# Predykcja na zbiorze testowym
predictions = mlp.predict(X_test)

# Sprawdzenie skuteczności
accuracy = accuracy_score(y_test, predictions)

print(f"Accuracy: {accuracy * 100:.2f}%")