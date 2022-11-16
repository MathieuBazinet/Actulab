import numpy as np
import pandas as pd
from Fairness_aware_model import FairnessAwareModel


protected_attributes = ['gender']

# Standard scaling for regression

dataCar = pd.read_csv("./dataCar.csv")
binary_answer = dataCar["clm"].values
numclaim = dataCar["numclaims"].values
reg_claim = dataCar["claimcst0"].values

dataCar = dataCar.drop(["X_OBSTAT_", "clm", "numclaims", "claimcst0"], axis=1)

# TODO encoder correctement les différentes valeurs qui ne sont pas des entiers
dataCar['gender'].replace(to_replace=np.unique(dataCar['gender']), inplace=True, value=range(np.unique(dataCar['gender']).shape[0]))
dataCar['veh_body'].replace(to_replace=np.unique(dataCar['veh_body']), inplace=True, value=range(np.unique(dataCar['veh_body']).shape[0]))
dataCar['area'].replace(to_replace=np.unique(dataCar['area']), inplace=True, value=range(np.unique(dataCar['area']).shape[0]))

protected_values = dataCar[protected_attributes].values

data = dataCar.drop(protected_attributes,axis=1).values

fam_poisson = FairnessAwareModel(regularization=100, protected_values=protected_values, family="poisson")
fam_poisson.fit(data, binary_answer)
# fam_poisson.fit(data, numclaim)
results = fam_poisson.predict(data)

# fam_gamma = FairnessAwareModel(regularization=100, protected_values=protected_values, family="gamma", alpha=2)
# fam_gamma.fit(data, reg_claim)
# fam_gamma.fit(data, reg_claim)
# # fam.fit(data, reg_claim)
# results = fam_gamma.predict(data)