import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

CollegeDF = pd.read_csv("vue_segpa.csv", sep=";")

CollegeDF['nbre_eleves_segpa'] = pd.to_numeric(CollegeDF['nbre_eleves_segpa'],
                                               errors='coerce')
CollegeDF = CollegeDF.dropna()

CollegeAr = CollegeDF.to_numpy()

mean_std = np.vstack([np.mean(CollegeAr, axis=0), np.std(CollegeAr, axis=0)])

print("Moyennes des colonnes : ", mean_std[0])
print("Écarts-types des colonnes : ", mean_std[1])

cov_matrix = CollegeDF.cov()

print("Matrice de Covariance :")
print(cov_matrix)


def centre_reduire(T):
                                               moyenne = np.mean(T, axis=0)
                                               ecart_type = np.std(T, axis=0)
                                               return (T -
                                                       moyenne) / ecart_type


CollegeAr_CR = centre_reduire(CollegeAr)

X = np.delete(CollegeAr_CR, 1, axis=1)
Y = CollegeAr_CR[:, 1]

# Régression linéaire multiple
linear_regression = LinearRegression()
linear_regression.fit(X, Y)
a = linear_regression.coef_

print("Coefficients de régression :", a)

Y_pred = linear_regression.predict(X)

print("Prédictions :", Y_pred)

# Coefficient de corrélation multiple
CorS = linear_regression.score(X, Y)
print("Coefficient de la corrélation multiple : ", CorS)


# Fonction pour créer un diagramme en bâtons
def diag_batons(colonne, titre="Diagramme en bâtons"):
                                               plt.figure()
                                               plt.hist(colonne,
                                                        bins=20,
                                                        edgecolor='black')
                                               plt.xlabel("Valeurs")
                                               plt.ylabel("Fréquence")
                                               plt.title(titre)
                                               plt.show()


diag_batons(CollegeDF['effectifs_garcons'],
            "Diagramme en bâtons de l'effectif de garçons")

CollegeDF['pourcentage_filles'] = CollegeDF['effectifs_filles'] / (
    CollegeDF['effectifs_filles'] + CollegeDF['effectifs_garcons']) * 100

# Créer un diagramme en bâtons pour le pourcentage de filles
diag_batons(CollegeDF['pourcentage_filles'],
            "Diagramme en bâtons du pourcentage de filles")
