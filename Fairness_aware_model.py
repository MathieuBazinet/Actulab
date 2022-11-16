import numpy as np
from scipy.optimize import minimize
from scipy.special import factorial
import pandas as pd
from sklearn.utils import check_random_state
import statsmodels.api as sm
import scipy.special


class FairnessAwareModel:

    def __init__(self, regularization, protected_values, offset=None, beta_init=None, family="binomial", alpha=None, seed=42):
        """
        La fonction de lien utilisée est le lien canonique. Pour régression poisson : lien log. Pour régression binomiale : logit.
        family (default="binomial") : "poisson" ou "binomial"
        """
        # On verra si on a besoin de random states mais je les mets là au cas où

        # On crée une liste d'attributs protégés
        self.protected_values = protected_values

        # On attribue un paramètre de régularisation par attribut protégé
        if not isinstance(regularization, list):
            self.regularization = [regularization] * self.protected_values.shape[1]
        else:
            if len(regularization) != self.protected_values.shape[1]:
                raise AssertionError("Le nombre de paramètres de régularisation " + \
                                     "n'est pas égal au nombre d'attributs protégés")
            self.regularization = regularization

        self.predict_on_subset = False
        self.random_state = check_random_state(seed)
        self.np_random_state = np.random.RandomState(seed)
        self.beta = None
        self.beta_init = beta_init
        #TODO : initialiser les betas aux valeurs du MLE?
        self.offset = offset
        self.family = family
        self.alpha = alpha
        
        #TODO : pas sûr que je vais m'en servir
        if self.family== "poisson":
            link="log"
            self.log_vraisemblance = self.log_vraisemblance_poisson
            self.penalization = self.equalized_odds_penalization
        elif self.family=="binomial":
            link="logit"
            self.log_vraisemblance = self.log_vraisemblance_binomial
            self.penalization = self.equalized_odds_penalization
        elif self.family == "gamma":
            self.log_vraisemblance = self.log_vraisemblance_gamma
            link="log"
            self.penalization = self.weak_actuarial_group_fairness
            if self.alpha is None:
                raise ValueError("Aucun gamma sélectionné pour une famille gamma.")
        self.link=link

    def beta_dot(self, X):
        if self.beta is None or self.offset is None:
            raise RuntimeError("The model was not fitted on the data.")

        # if self.predict_on_subset:
        #     offset = np.log(self.offset[self.subset])
        # else:
        #     offset = np.log(self.offset)
        return  X @ self.beta #+ offset

    def log_vraisemblance_poisson(self, X, y):
        lin_predictor = self.beta_dot(X).reshape(-1, 1)
        y = y.reshape(-1, 1)
        yx = np.sum(y * lin_predictor - np.exp(lin_predictor))
        c = np.sum(-np.log(factorial(y)))
        return yx + c
    
    def log_vraisemblance_binomial(self, X, y):
        # Diapos 35/50 chapitre 2
        lin_predictor = self.beta_dot(X).reshape(-1, 1)
        pi = 1/(1+np.exp(-lin_predictor))
        #return np.sum(y.reshape(-1,1) * (np.log(pi) - np.log(1-pi)) + np.log(1-pi) )
        #return np.sum(y.reshape(-1,1) * lin_predictor - lin_predictor - np.log(1+np.exp(-lin_predictor)))
        return np.sum(y.reshape(-1,1) * lin_predictor - np.log(1+np.exp(lin_predictor)))
        #return np.sum(np.log(y.reshape(-1,1) * pi + (1-y).reshape(-1,1)*(1-pi))) # https://matthew-brett.github.io/cfd2020/more-regression/logistic_convexity.html

    def log_vraisemblance_gamma(self, X, y):
        lin_predictor = self.beta_dot(X).reshape(-1, 1)
        y = y.reshape(-1,1)
        logy = np.log(y) * (self.alpha - 1)
        if self.link == "log":
            mu = np.exp(lin_predictor)
        else:
            mu = 1/lin_predictor
        yx = -self.alpha * (y / mu + np.log(mu))
        return np.sum(yx + logy)

    def no_penalization_loss(self, beta):

        self.beta = beta

        if self.family == "poisson":
            return self.log_vraisemblance_poisson(self.X, self.y)
        elif self.family=="binomial":
            return self.log_vraisemblance_binomial(self.X, self.y)
        elif self.family == "gamma":
            return self.log_vraisemblance_gamma(self.X, self.y)
        else:
            raise NotImplementedError


    def equalized_odds_penalization(self, beta):
        """
        Idée : probabilité que \hat{Y} = 1 sachant A=a (equalized odds)
        :param beta:
        :return:
        """
        self.beta = beta
        self.beta = self.beta / np.linalg.norm(self.beta)
        print(self.beta)
        log_vraisemblance = self.log_vraisemblance(self.X, self.y)
        loss = 0
        self.predict_on_subset = True
        for s_index in range(self.protected_values.shape[1]):
            values = np.unique(self.protected_values[:, s_index])
            regularization_parameter = self.regularization[s_index]
            y_values = list(np.unique(self.y))
            for y_val in y_values:
                predict_list = []
                new_y = np.array([int(y_val == i) for i in self.y]).reshape(-1, 1)
                for v in values:
                    self.subset = np.where(self.protected_values[:, s_index] == v)[0]
                    if int(np.sum(new_y[self.subset])) != 0:
                        predict_list.append(np.sum(
                            self.predict(self.X[self.subset, :]) * new_y[self.subset].reshape(-1, 1))
                                            / np.sum(new_y[self.subset]))
                for i in range(len(predict_list)):
                    for j in range(i+1, len(predict_list)):
                        loss += regularization_parameter * np.abs(predict_list[i] - predict_list[j])
        self.predict_on_subset = False
        return -log_vraisemblance + loss

    def weak_actuarial_group_fairness(self, beta):
        self.beta = beta
        print(self.beta)
        log_vraisemblance = self.log_vraisemblance(self.X, self.y)
        loss = 0
        self.predict_on_subset = True
        for s_index in range(self.protected_values.shape[1]):
            values = np.unique(self.protected_values[:, s_index])
            regularization_parameter = self.regularization[s_index]
            predict_list = []
            row_index = list(np.quantile(self.y, [0,0.25,0.5,0.75,1]))
            for v in values:
                somme = 0
                for i in range(len(row_index)-1):
                    interval = set(list(np.where(self.y > row_index[i])[0])) & set(list(np.where(self.y < row_index[i+1])[0]))
                    self.subset = np.where(self.protected_values[list(interval), s_index] == v)[0]
                    somme += np.sum(
                        self.predict(self.X[self.subset, :]) * self.y[self.subset].reshape(-1,1)) / np.sum(self.y[self.subset])
                predict_list.append(somme)
            for i in range(len(predict_list)):
                for j in range(i+1, len(predict_list)):
                    loss += regularization_parameter * np.abs(predict_list[i] - predict_list[j])
        self.predict_on_subset = False
        return -log_vraisemblance + loss

    def fit(self, X_train, y_train):

        if self.family == "gamma":
            index = np.where(y_train > 0)[0]
            self.X = X_train[index, :]
            self.y = y_train[index]
        else:
            self.X = X_train
            self.y = y_train

        if self.beta_init is None:
            self.beta_init = np.ones(self.X.shape[1]) / self.X.shape[1]
            self.beta_init = np.random.rand(X_train.shape[1])
            ## essai de "warm start"...
            reference_model = sm.Logit(y_train, X_train).fit()
            self.beta_init = reference_model.params
            print(self.beta_init)

        if self.offset is None:
            self.offset = np.ones(self.X.shape[0])

        res = minimize(self.penalization, self.beta_init, method='BFGS', options={'maxiter': 1000, 'disp': True})
        self.beta = res.x
        self.beta_init = self.beta

    def predict(self, X, type="response"):
        """Prédiction de la variable réponse pour une matrice X donnée.

        Args:
            X (np.array): matrice des données en format one-hot
            type (str, optional): Retourne la prédiction. "response" pour la probabilité. "value" pour 1/0 (si logistique), "linear" pour le prédicteur linéaire (B^tX) Defaults to "response".
        """
        lin_predictor = self.beta_dot(X).reshape(-1, 1)

        if type=="linear":
            prediction = lin_predictor
        else:
            if self.family == "binomial":
                prediction = 1/(1+np.exp(-lin_predictor))
            elif self.family == "poisson":
                prediction = np.exp(lin_predictor)
            elif self.family == "gamma":
                if self.link == "log":
                    prediction = np.exp(lin_predictor)
                else:
                    prediction = 1/lin_predictor
            else:
                raise NotImplementedError

        return prediction

if __name__ == "__main__":
    print("hello world")
    #TODO : comparer l'output de la régression logistique avec un modèle de statsmodel