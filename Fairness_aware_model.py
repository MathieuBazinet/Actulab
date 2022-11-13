import numpy as np
from scipy.optimize import minimize
from scipy.special import factorial
import pandas as pd
from sklearn.utils import check_random_state


class FairnessAwareModel:

    def __init__(self, regularization, protected_attributes, offset=None, beta_init=None,  seed=42):
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

    def penalized_loss_1(self, beta):
        """
        Idée : minimiser x_i^{A=a} - x_i^{A=b}
        Problème : Solution "optimale" donnerait simplement beta_A = 0
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
        if self.offset is None:
            self.offset = np.ones(X_train.shape[0])
        res = minimize(self.penalized_loss_1, self.beta_init, method='BFGS', options={'maxiter': 500})
        self.beta = res.x
        self.beta = self.beta / np.linalg.norm(self.beta)
        self.beta_init = self.beta


if __name__ == "__main__":
    print("hello world")