from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Pobranie zbioru danych
digits = datasets.load_digits()

# Podział na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=42)

# Definicja różnych kombinacji parametrów
parameters = [
    {'hidden_layer_sizes': (100,), 'activation': 'relu', 'solver': 'adam', 'max_iter': 1000},
    {'hidden_layer_sizes': (50,50), 'activation': 'tanh', 'solver': 'sgd', 'max_iter': 1000, 'learning_rate_init': 0.01},
    {'hidden_layer_sizes': (30,30,30), 'activation': 'logistic', 'solver': 'sgd', 'max_iter': 1000, 'learning_rate_init': 0.1},
    {'hidden_layer_sizes': (10,10,10,10), 'activation': 'relu', 'solver': 'adam', 'max_iter': 1000, 'learning_rate_init': 0.001},
    {'hidden_layer_sizes': (10,10,10,10), 'activation': 'tanh', 'solver': 'adam', 'max_iter': 10000, 'learning_rate_init': 0.0001}
]

for params in parameters:
    # Stworzenie sieci neuronowej z danymi parametrami
    mlp = MLPClassifier(**params, random_state=42)

    # Trening sieci
    mlp.fit(X_train, y_train)

    # Predykcja na zbiorze testowym
    predictions = mlp.predict(X_test)

    # Obliczenie skuteczności
    accuracy = accuracy_score(y_test, predictions)

    print(f"Parameters: {params}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))
    print("\n")