import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import time

# Pobranie i wczytanie zbioru danych
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/yeast/yeast.data"
names = ['Sequence Name', 'mcg', 'gvh', 'alm', 'mit', 'erl', 'pox', 'vac', 'nuc', 'class']
df = pd.read_csv(url, names=names, delim_whitespace=True)

# Wstępna analiza danych
print(df.describe())

# Zamiana etykiet tekstowych na liczbowe
le = preprocessing.LabelEncoder()
df['class'] = le.fit_transform(df['class'])

# Podział na zbiór treningowy i testowy
X = df.drop(['Sequence Name', 'class'], axis=1)
y = df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Stworzenie sieci neuronowej
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=3000, random_state=1)

# Trening sieci i pomiar czasu
start_time = time.time()
mlp.fit(X_train, y_train)
end_time = time.time()

# Predykcja na zbiorze testowym
predictions = mlp.predict(X_test)

# Obliczenie skuteczności
accuracy = accuracy_score(y_test, predictions)

print(f"Training time: {end_time - start_time:.2f}s")
print(f"Accuracy: {accuracy * 100:.2f}%")
# Macierz na zbiorze testowym
print("Confusion matrix:")
print(confusion_matrix(y_test, predictions))
# Macierz na zbiorze treningowym
print("Confusion matrix (train set):")
print(confusion_matrix(y_train, mlp.predict(X_train)))