import numpy as np
import scipy.optimize
import pandas as pd
from sklearn.utils import check_random_state

class FairnessAwareModel:

    def __init__(self, regularization, link_function, seed=42):
        self.reg = regularization
        self.link_function = link_function
        # On verra si on a besoin de random states mais je les mets là au cas où
        self.random_state = check_random_state(seed)
        self.np_random_state = np.random.RandomState(seed)

    def predict(self):
        pass

    def fit(self, X_train, y_train):
        pass

    def custom_loss(self):
        pass


if __name__ == "__main__":
    print("hello world")