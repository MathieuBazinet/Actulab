import numpy as np
import pandas as pd
from Fairness_aware_model import FairnessAwareModel
from os.path import join, dirname, abspath, isdir, isfile

def hot_encoder(data, optional_columns):
    categorical_cols = []
    for column in data.columns:
            if data[column].dtype == object or column in optional_columns:
                categorical_cols.append(column)

    to_return = pd.get_dummies(data, columns = categorical_cols)
    return to_return


if __name__ == "__main__":
    protected_attributes = ['gender', 'agecat']
    family = "poisson"
    cross_val = False
    # Standard scaling for regression

    dataCar = pd.read_csv("./dataCar_clean.csv")
    train = dataCar.loc[dataCar['train'] == 1]
    test = dataCar.loc[dataCar['train'] == 0]

    protected_values = train[protected_attributes].values
    train = train.drop(protected_attributes, axis=1)
    test = test.drop(protected_attributes, axis=1)

    train_encoded = hot_encoder(train, [])
    test_encoded = hot_encoder(test, [])

    clm_train = train["clm"].values
    numclaim_train = train["numclaims"].values
    reg_claim_train = train["claimcst0"].values

    clm_test = test["clm"].values
    numclaim_test = test["numclaims"].values
    reg_claim_test = test["claimcst0"].values

    borne_clm = np.mean(clm_train)
    born_numclaim = np.mean(numclaim_train)

    train_encoded = train_encoded.drop(
        ["clm", "numclaims", "claimcst0", "veh_body_BUS", "area_A", "exposure", "train"], axis=1).values
    test_encoded = test_encoded.drop(
        ["clm", "numclaims", "claimcst0", "veh_body_BUS", "area_A", "exposure", "train"], axis=1).values

    regs = np.logspace(-2, 5, 8) if cross_val else np.array([100])
    # TODO Si tu change les chiffres dans le logspace, le dernier chiffre va être (top value) - (min value) + 1.
    #  Par exemple, 5 - (-2) + 1 = 8
    results = np.zeros((test_encoded.shape[0], regs.shape[0]))
    index = 0
    error = []
    if family == "logistic":
        for reg in regs:
            fam_logistic = FairnessAwareModel(regularization=reg, protected_values=protected_values, family="binomial")
            fam_logistic.fit(train_encoded, clm_train,warm_start=True)
            results[:, index] = fam_logistic.predict(test_encoded).reshape(-1,)
            index += 1
    elif family == "poisson":
        for reg in regs:
            fam_poisson = FairnessAwareModel(regularization=reg, protected_values=protected_values, family="poisson")
            fam_poisson.fit(train_encoded, clm_train, warm_start=True)
            results[:, index] = fam_poisson.predict(test_encoded).reshape(-1,)
            index += 1
    elif family == "gamma":
        for reg in regs:
            fam_clm = FairnessAwareModel(regularization=reg, protected_values=protected_values, family="binomial")
            fam_clm.fit(train_encoded, clm_train, warm_start=True)

            fam_gamma = FairnessAwareModel(regularization=reg, protected_values=protected_values, family="gamma",
                                           alpha=(1/0.3429485))
            fam_gamma.fit(train_encoded, reg_claim_train, warm_start=True)

            results[:, index] = (fam_gamma.predict(test_encoded) * fam_clm.predict(test_encoded)).reshape(-1,)
            index += 1

    path = join(dirname(abspath(__file__)), f"results_crossval_{family}.csv")
    pd.DataFrame(results).to_csv(path, index=False)




