import numpy as np
from scipy.optimize import minimize
from scipy.special import factorial
try:
    warm_start_possible = True
    import statsmodels.api as sm
except:
    warm_start_possible = False


class FairnessAwareModel:
    """
    Modèle linéaire généralisé dont la log-vraisemblance est pénalisé par une mesure d'équité
    """

    def __init__(self, regularization, protected_values, beta_init=None, family="binomial",
                 equity_metric="EO", alpha=None):
        """
        La fonction de lien utilisée est le lien canonique. Pour régression poisson et gamma : lien log.
        Pour régression binomiale : logit.

        :param regularization: Le paramètre de régularisation qui indique l'impact de la pénalisation d'équité
        :param protected_values: Le tableau contenant les valeurs protégés associées aux données d'entraînement
        :param beta_init: Les valeurs d'initialisations des beta
        :param family: La famille de distribution
        :param equity_metric: La métrique d'équité à utiliser pour la régression logistique
                                (default="EO"): EO : equalized odds. DP = demographic parity.
        :param alpha: Le paramètre de shape pour la régression gamma
        """
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

        self.beta = None
        self.beta_init = beta_init
        self.family = family
        self.alpha = alpha

        if self.family== "poisson":
            self.log_vraisemblance = self.log_vraisemblance_poisson
            self.penalization = self.equalized_odds_penalization
        elif self.family=="binomial":
            self.log_vraisemblance = self.log_vraisemblance_binomial
            if equity_metric == "EO":
                self.penalization = self.equalized_odds_penalization
            elif equity_metric == "DP":
                self.penalization = self.demographic_parity_penalization
            else:
                raise ValueError("Pénalité non valide pour la régression logistique.")
        elif self.family == "gamma":
            self.log_vraisemblance = self.log_vraisemblance_gamma
            self.penalization = self.actulab_group_fairness
            if self.alpha is None:
                raise ValueError("Aucun gamma sélectionné pour une famille gamma.")

    def beta_dot(self, X):
        """
        Fonction privée qui ne sert que lorsqu'on calcule la vraisemblance
        :param X: Les données qu'on veut multiplier avec les betas.
        :return:
        """
        if self.beta is None:
            raise RuntimeError("The model was not fitted on the data.")
        return X @ self.beta

    def log_vraisemblance_poisson(self, X, y):
        """
        Calcul de la log-vraisemblance de la régression Poisson
        :param X: Données d'entraînement
        :param y: Variable réponse
        :return: La log-vraisemblance
        """
        lin_predictor = self.beta_dot(X).reshape(-1, 1)
        y = y.reshape(-1, 1)
        yx = np.sum(y * lin_predictor - np.exp(lin_predictor))
        c = np.sum(-np.log(factorial(y)))
        return yx + c
    
    def log_vraisemblance_binomial(self, X, y):
        """
        Calcul de la log-vraisemblance de la régression binomiale
        :param X: Données d'entraînement
        :param y: Variable réponse
        :return: La log-vraisemblance
        """
        lin_predictor = self.beta_dot(X).reshape(-1, 1)
        return np.sum(y.reshape(-1,1) * lin_predictor - np.log(1+np.exp(lin_predictor)))

    def log_vraisemblance_gamma(self, X, y):
        """
        Calcul de la log-vraisemblance de la régression gamma
        :param X: Données d'entraînement
        :param y: Variable réponse
        :return: La log-vraisemblance
        """
        lin_predictor = self.beta_dot(X).reshape(-1, 1)
        y = y.reshape(-1,1)
        logy = np.log(y) * (self.alpha - 1)
        mu = np.exp(lin_predictor)
        yx = -self.alpha * (y / mu + np.log(mu))
        return np.sum(yx + logy)

    def no_penalization_loss(self, beta):
        """
        Mesure qui ne pénalise pas du tout la perte.
        :param beta:
        :return:
        """
        self.beta = beta

        if self.family == "poisson":
            return self.log_vraisemblance_poisson(self.X, self.y)
        elif self.family == "binomial":
            return self.log_vraisemblance_binomial(self.X, self.y)
        elif self.family == "gamma":
            return self.log_vraisemblance_gamma(self.X, self.y)
        else:
            raise NotImplementedError

    def demographic_parity_penalization(self, beta):
        """
        Implémentation de la parité démographique
        :param beta: Les paramètres à optimiser
        :return: la valeur à minimiser
        """
        self.beta = beta
        log_vraisemblance = self.log_vraisemblance(self.X, self.y)
        loss = 0
        # Calculer la parité pour chaque attribut protégé
        for s_index in range(self.protected_values.shape[1]):
            values = np.unique(self.protected_values[:, s_index])
            regularization_parameter = self.regularization[s_index]
            predict_list = []
            # Calculer la parité pour chaque valeur que peut prendre l'attribut protégé
            for v in values:
                self.subset = np.where(self.protected_values[:, s_index] == v)[0]
                predict_list.append(np.sum(self.predict(self.X[self.subset, :]))/ self.subset.shape[0])
            # Calculer la différence entre la parité pour différents attributs
            for i in range(len(predict_list)):
                for j in range(i+1, len(predict_list)):
                    loss += regularization_parameter * np.abs(predict_list[i] - predict_list[j])
        return -log_vraisemblance + loss


    def equalized_odds_penalization(self, beta):
        """
        Implémentation de la disparité moindres
        :param beta:
        :return:
        """
        self.beta = beta
        log_vraisemblance = self.log_vraisemblance(self.X, self.y)
        loss = 0
        # Calculer la parité pour chaque attribut protégé
        for s_index in range(self.protected_values.shape[1]):
            values = np.unique(self.protected_values[:, s_index])
            regularization_parameter = self.regularization[s_index]
            y_values = list(np.unique(self.y))
            # Calculer la parité pour chaque valeur que y peut prendre
            for y_val in y_values:
                predict_list = []
                new_y = np.array([int(y_val == i) for i in self.y]).reshape(-1, 1)
                # Calculer la parité pour chaque valeur que peut prendre l'attribut protégé
                for v in values:
                    self.subset = np.where(self.protected_values[:, s_index] == v)[0]
                    if int(np.sum(new_y[self.subset])) != 0:
                        predict_list.append(np.sum(
                            self.predict(self.X[self.subset, :]) * new_y[self.subset].reshape(-1, 1))
                                            / np.sum(new_y[self.subset]))
                # Calculer la différence entre la parité pour différents attributs
                for i in range(len(predict_list)):
                    for j in range(i+1, len(predict_list)):
                        loss += regularization_parameter * np.abs(predict_list[i] - predict_list[j])
        return -log_vraisemblance + loss

    def actulab_group_fairness(self, beta):
        """
        Disparité de la parité actulab
        :param beta:
        :return:
        """
        self.beta = beta
        print(self.beta)
        log_vraisemblance = self.log_vraisemblance(self.X, self.y)
        loss = 0
        # Calculer la parité pour chaque attribut protégé
        for s_index in range(self.protected_values.shape[1]):
            values = np.unique(self.protected_values[:, s_index])
            regularization_parameter = self.regularization[s_index]
            predict_list = []
            row_index = list(np.quantile(self.y, [0,0.25,0.5,0.75,1]))
            # Calculer la parité pour chaque valeur que peut prendre l'attribut protégé
            for v in values:
                somme = 0
                # Calculer la parité pour chaque groupe de risque
                for i in range(len(row_index)-1):
                    interval = set(list(np.where(self.y > row_index[i])[0])) & set(list(np.where(self.y < row_index[i+1])[0]))
                    self.subset = np.where(self.protected_values[list(interval), s_index] == v)[0]
                    somme += np.mean(self.predict(self.X[self.subset, :])) / (row_index[i+1]-row_index[i])
                predict_list.append(somme)
            # Calculer la différence entre la parité pour différents attributs
            for i in range(len(predict_list)):
                for j in range(i+1, len(predict_list)):
                    loss += regularization_parameter * np.abs(predict_list[i] - predict_list[j])
        return -log_vraisemblance + loss

    def fit(self, X_train, y_train, warm_start=False):

        if self.family == "gamma":
            index = np.where(y_train > 0)[0]
            self.X = X_train[index, :]
            self.y = y_train[index]
        else:
            self.X = X_train
            self.y = y_train

        if self.beta_init is None:
            if warm_start and warm_start_possible:
                if self.family == "binomial":
                    reference_model = sm.Logit(y_train, X_train).fit()
                    self.beta_init = reference_model.params
                elif self.family == "poisson":
                    reference_model = sm.GLM(y_train, X_train, family=sm.families.Poisson()).fit()
                    self.beta_init = reference_model.params
                elif self.family == "gamma":
                    reference_model = sm.GLM(y_train, X_train, family=sm.families.Gamma()).fit()
                    self.beta_init = reference_model.params
            else:
                self.beta_init = np.ones(self.X.shape[1]) / self.X.shape[1]

        res = minimize(self.penalization, self.beta_init, method='BFGS', options={'maxiter': 500, 'disp': True})
        self.beta = res.x
        self.beta_init = self.beta

    def predict(self, X, type="response"):
        """Prédiction de la variable réponse pour une matrice X donnée.

        Args:
            X (np.array): matrice des données en format one-hot
            type (str, optional): Retourne la prédiction. "response" pour la probabilité.
             "value" pour 1/0 (si logistique), "linear" pour le prédicteur linéaire (B^tX) Defaults to "response".
        """
        lin_predictor = self.beta_dot(X).reshape(-1, 1)

        if type=="linear":
            prediction = lin_predictor
        elif self.family == "binomial":
            prediction = 1/(1+np.exp(-lin_predictor))
        elif self.family == "poisson":
            prediction = np.exp(lin_predictor)
        elif self.family == "gamma":
            prediction = np.exp(lin_predictor)
        else:
            raise NotImplementedError

        return prediction

if __name__ == "__main__":
    print("hello world")