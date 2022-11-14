import numpy as np
from scipy.optimize import minimize
from scipy.special import factorial
import pandas as pd
from sklearn.utils import check_random_state
import statsmodels.api as sm


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
        #TODO : initialiser les betas aux valeurs du MLE?
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
        return  X @ self.beta + np.log(self.offset) # equivalent de faire values

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

    def penalized_loss_1(self, beta):
        """
        Idée : minimiser x_i^{A=a} - x_i^{A=b}
        Problème : Solution "optimale" donnerait simplement beta_A = 0
        :param beta:
        :return:
        """
        self.beta = beta
        #self.beta = self.beta / np.linalg.norm(self.beta)

        if self.family=="poisson":
            log_vraisemblance = self.log_vraisemblance_poisson(self.X, self.y)
        elif self.family=="binomial":
            log_vraisemblance = self.log_vraisemblance_binomial(self.X, self.y)
        else:
            raise NotImplementedError

        loss = 0
        ### je commente pour simplement voir si la vraisemblance sans pénalisation est ok
        # for s_index in range(len(self.protected_attributes)):
        #     s = self.protected_attributes[s_index]
        #     predict_list = []
        #     regularization_parameter = self.regularization[s_index]
        #     for a in np.unique(self.X[:, s]):
        #         X_a = self.X
        #         X_a[:, s] = a
        #         predict_list.append(self.predict(X_a, type="response"))
        #     for i in range(len(predict_list)):
        #         for j in range(i, len(predict_list)):
        #             loss += regularization_parameter * np.sum((predict_list[i] - predict_list[j])**2) 
        return -1*log_vraisemblance + loss

    def penalized_loss_2(self, beta):
        """
        Idée : probabilité que \hat{Y} = 1 sachant A=a (equalized odds)
        :param beta:
        :return:
        """
        self.beta = beta
        self.beta = self.beta / np.linalg.norm(self.beta)
        print(self.beta)
        log_vraisemblance = self.log_vraisemblance_poisson(self.X, self.y)
        loss = 0
        for s_index in range(len(self.protected_attributes)):
            s = self.protected_attributes[s_index]
            values = np.unique(self.X[:, s_index])
            regularization_parameter = self.regularization[s_index]
            predict_list = []
            for v in values:
                index = np.where(self.X[:,s_index] == v)[0]
                predict_list.append(np.sum(self.predict(self.X[:, index]) * self.y[index]) / np.sum(self.y[index]))
            for i in range(len(predict_list)):
                for j in range(i, len(predict_list)):
                    loss += regularization_parameter * np.sum((predict_list[i] - predict_list[j])**2)
        return -log_vraisemblance + loss

    def fit(self, X_train, y_train):

        self.X = X_train
        self.y = y_train

        if self.beta_init is None:
            self.beta_init = np.ones(X_train.shape[1]) / X_train.shape[1]
            #self.beta_init = np.random.rand(X_train.shape[1])
            ## essai de "warm start"...
            #reference_model = sm.Logit(y_train, X_train).fit()
            #self.beta_init = reference_model.params

        if self.offset is None:
            self.offset = np.ones(X_train.shape[0])

        res = minimize(self.penalized_loss_1, self.beta_init, method='BFGS', options={'maxiter': 500})
        res = minimize(self.penalized_loss, self.beta_init, method='BFGS', options={'maxiter': 500, 'disp': True})
        self.beta = res.x
        #self.beta = self.beta / np.linalg.norm(self.beta)
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
        
        return prediction

if __name__ == "__main__":
    print("hello world")
    #TODO : comparer l'output de la régression logistique avec un modèle de statsmodel