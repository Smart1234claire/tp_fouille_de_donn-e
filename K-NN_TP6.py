import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# Chargement des données depuis un fichier CSV
data = pd.read_csv('C:/Users/22890/Documents/MASTER1/FD/Social_Network_Ads.csv')  # Assurez-vous de remplacer 'votre_fichier.csv' par le chemin vers votre propre fichier de données

# Séparation des variables indépendantes (X) et de la variable cible (y)
X = data[['Age', 'EstimatedSalary']]
y = data['Purchased']

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalisation des données
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Création du modèle KNN
k = 5  # Nombre de voisins
knn_model = KNeighborsClassifier(n_neighbors=k)

# Entraînement du modèle
knn_model.fit(X_train, y_train)

# Prédiction sur l'ensemble de test
y_pred = knn_model.predict(X_test)

# Séparation des données en utilisateurs qui ont acheté et ceux qui n'ont pas acheté
purchased_users = data[data['Purchased'] == 1]
not_purchased_users = data[data['Purchased'] == 0]

# Création du nuage de points
plt.figure(figsize=(10, 6))
plt.scatter(purchased_users['Age'], purchased_users['EstimatedSalary'], color='green', label='Purchased')
plt.scatter(not_purchased_users['Age'], not_purchased_users['EstimatedSalary'], color='red', label='Not Purchased')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.title('Purchased vs. Not Purchased Users by Age and Estimated Salary')
plt.legend()
plt.grid(True)
plt.show()




