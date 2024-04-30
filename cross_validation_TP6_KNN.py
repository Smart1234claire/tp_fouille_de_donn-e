import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt

# Charger le dataset depuis un fichier CSV
dataset = pd.read_csv('C:/Users/22890/Documents/MASTER1/FD/Social_Network_Ads.csv')

# Séparation des caractéristiques (X) et de la variable cible (y)
X = dataset[['Gender', 'Age', 'EstimatedSalary']]
y = dataset['Purchased']

# Encoder les variables catégorielles (si nécessaire)
X = pd.get_dummies(X)

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Mise à l'échelle des caractéristiques
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Liste des valeurs de k à tester
k_values = list(range(1, 21))  # Testons pour k de 1 à 20

# Stockage des scores de validation croisée pour chaque valeur de k
cv_scores = []

# Boucle sur les valeurs de k
for k in k_values:
    # Initialiser et entraîner le modèle KNN
    classifier = KNeighborsClassifier(n_neighbors=k)

    # Calculer les scores de validation croisée
    scores = cross_val_score(classifier, X_train, y_train, cv=5)

    # Stocker la moyenne des scores de validation croisée pour cette valeur de k
    cv_scores.append(scores.mean())

# Choisir la meilleure valeur de k avec le score de validation croisée le plus élevé
best_k = k_values[np.argmax(cv_scores)]
print("Meilleure valeur de k :", best_k)

# Entraîner le modèle KNN avec la meilleure valeur de k
best_classifier = KNeighborsClassifier(n_neighbors=best_k)
best_classifier.fit(X_train, y_train)

# Évaluer les performances du modèle sur l'ensemble de test
test_accuracy = best_classifier.score(X_test, y_test)
print("Précision sur l'ensemble de test avec la meilleure valeur de k :", test_accuracy)

# Tracer les scores de validation croisée en fonction de k
plt.plot(k_values, cv_scores, marker='o')
plt.xlabel('Nombre de voisins (k)')
plt.ylabel('Score de validation croisée')
plt.title('Scores de validation croisée pour différentes valeurs de k')
plt.grid(True)
plt.show()
