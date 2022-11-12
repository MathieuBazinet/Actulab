import numpy as np
from scipy.optimize import minimize
from scipy.special import factorial
import pandas as pd
from sklearn.utils import check_random_state


class FairnessAwareModel:

    def __init__(self, regularization, protected_attributes, offset=None, beta_init=None, family="binomial", seed=42):
        """
        La fonction de lien utilisée est le lien canonique. Pour régression poisson : lien log. Pour régression binomiale : logit.
        family (default="binomial") : "poisson" ou "binomial"
        """
        # On verra si on a besoin de random states mais je les mets là au cas où

        # On crée une liste d'attributs protégés
        if not isinstance(protected_attributes, list):
            self.protected_attributes = [protected_attributes]
        else:
            self.protected_attributes = protected_attributes

        # On attribue un paramètre de régularisation par attribut protégé
        if not isinstance(regularization, list):
            self.regularization = [regularization] * len(self.protected_attributes)
        else:
            if len(regularization) != len(self.protected_attributes):
                raise AssertionError("Le nombre de paramètres de régularisation " + \
                                     "n'est pas égal au nombre d'attributs protégés")
            self.regularization = regularization

        self.random_state = check_random_state(seed)
        self.np_random_state = np.random.RandomState(seed)
        self.beta = None
        self.beta_init = beta_init
        self.offset = offset
        self.family = family
        
        #TODO : pas sûr que je vais m'en servir
        if self.family=="poisson":
            link="log"
        elif self.family=="binomial":
            link="logit"
        self.link=link

    def beta_dot(self, X):
        if self.beta is None or self.offset is None:
            raise RuntimeError("The model was not fitted on the data.")
        return X @ self.beta + np.log(self.offset)

    def predict(self, X):
        return np.exp(self.beta_dot(X))

    def log_vraisemblance_poisson(self, X, y):
        predicted_x = self.beta_dot(X).reshape(-1, 1)
        y = y.reshape(-1, 1)
        yx = np.sum(y * predicted_x - np.exp(predicted_x))
        c = np.sum(-np.log(factorial(y)))
        return yx + c
    
    def log_vraisemblance_binomial(self, X, y_):
        # Diapos 35/50 chapitre 2
        predicted_x = self.beta_dot(X).reshape(-1, 1)
        pi = 1/(1+exp(-1*predicted_x))
        return np.sum(y * (np.log(pi) - np.log(1-pi)) + np.log(1-pi) )

    def penalized_loss(self, beta):
        self.beta = beta
        self.beta = self.beta / np.linalg.norm(self.beta)
        print(self.beta)

        if self.family=="poisson":
            log_vraisemblance = self.log_vraisemblance_poisson(self.X, self.y)
        elif self.family=="binomial":
            log_vraisemblance = self.log_vraisemblance_binomial(self.X, self.y)
        else:
            raise NotImplementedError

        loss = 0
        for s_index in range(len(self.protected_attributes)):
            s = self.protected_attributes[s_index]
            predict_list = []
            regularization_parameter = self.regularization[s_index]
            for a in np.unique(self.X[:, s]):
                X_a = self.X
                X_a[:, s] = a
                predict_list.append(self.predict(X_a))
            for i in range(len(predict_list)):
                for j in range(i, len(predict_list)):
                    loss += regularization_parameter * np.sum((predict_list[i] - predict_list[j])**2)
        return -log_vraisemblance + loss

    def fit(self, X_train, y_train):
        self.X = X_train
        self.y = y_train
        if self.beta_init is None:
            self.beta_init = np.ones(X_train.shape[1]) / X_train.shape[1]
        if self.offset is None:
            self.offset = np.ones(X_train.shape[0])
        res = minimize(self.penalized_loss, self.beta_init, method='BFGS', options={'maxiter': 500})

        #TODO : QUESTION : pourquoi on normalise les betas? 
        # Je comprends que d'un point de vue numérique c'est fait dans les réseaux de neurones, mais si on fait ça 
        # les betas en sortie ne seront pas les bons (interprétation ne sera pas correcte)...
        # statsmodel utilise la fonction scipy.opimize.fmin_bfgs (qui revient à utiliser scipy.optimize.minimize(method="BFGS")), 
        # mais je ne pense pas qu'ils font une division (https://github.com/statsmodels/statsmodels/blob/main/statsmodels/base/optimizer.py#L478)
        self.beta = res.x
        self.beta = self.beta / np.linalg.norm(self.beta)
        self.beta_init = self.beta

    def predict(self, X, type="response"):
        """Prédiction de la variable réponse pour une matrice X donnée.

        Args:
            X (np.array): matrice des données en format one-hot
            type (str, optional): Retourne la prédiction. "response" pour la probabilité. "value" pour 1/0 (si logistique), "linear" pour le prédicteur linéaire (B^tX) Defaults to "response".
        """
        predicted_x = self.beta_dot(X).reshape(-1, 1)

        if type=="linear":
            prediction = predicted_x
        else:
            prediction = 1/(1+np.exp(-predicted_x))
        
        return prediction

if __name__ == "__main__":
    print("hello world")
    #TODO : vérifier que le code fonctionne (régression logistique)
    #TODO : comparer l'output de la régression logistique avec un modèle de statsmodel