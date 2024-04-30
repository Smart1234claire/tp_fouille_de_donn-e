import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

# Charger les ensembles de données d'entraînement et de test
train_data = pd.read_csv('RT_code_competition/train.csv')
test_data = pd.read_csv('RT_code_competition/test.csv')

# Exclure les colonnes non numériques
numeric_cols = train_data.select_dtypes(include=['int64', 'float64']).columns

# Remplacer les valeurs manquantes par la moyenne de chaque colonne numérique
train_data[numeric_cols] = train_data[numeric_cols].fillna(train_data[numeric_cols].mean())
test_data[numeric_cols] = test_data[numeric_cols].fillna(test_data[numeric_cols].mean())

# Supposons que vous avez déjà effectué le prétraitement et l'ingénierie des fonctionnalités
# Pour la démonstration, supprimons les colonnes non numériques
non_numeric_cols = train_data.select_dtypes(exclude=['int64', 'float64']).columns
X_train = train_data.drop(['SalePrice'] + list(non_numeric_cols), axis=1)
y_train = train_data['SalePrice']

# Encodage one-hot des variables catégorielles
X_train = pd.get_dummies(X_train)

# Entraîner votre modèle (Arbre de décision de régression) avec une profondeur maximale de 6
model = DecisionTreeRegressor(max_depth=4, min_samples_leaf=5)
model.fit(X_train, y_train)

# Visualiser l'arbre de décision
plt.figure(figsize=(20,10))
from sklearn.tree import plot_tree
plot_tree(model, filled=True, fontsize=8)
plt.show()
