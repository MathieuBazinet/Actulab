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
    family = "binomial"
    protected_attributes = ['gender'] 
    cross_val = True
    # Standard scaling for regression

    dataCar = pd.read_csv("./dataCar_clean.csv")
    train = dataCar.loc[dataCar['which_set'] == 0]
    test = dataCar.loc[dataCar['which_set'] != 0] # Ceci est les données de validation ET de test

    protected_values = train[protected_attributes].values
    train = train.drop(protected_attributes, axis=1)
    test = test.drop(protected_attributes, axis=1)
    train = train.drop("agecat", axis=1)# retirer agecat du dataset au début pour voir ce qui se passe avec gender "indépendamment" de la pénalisation sur agecat
    test = test.drop("agecat", axis=1)

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
        ["clm", "numclaims", "claimcst0", "veh_body_BUS", "area_A", "exposure", "which_set"], axis=1).values
    test_encoded = test_encoded.drop(
        ["clm", "numclaims", "claimcst0", "veh_body_BUS", "area_A", "exposure", "which_set"], axis=1).values

    #np.logspace(-2, 4, 15)
    regs = np.logspace(-2, 5, 50) if cross_val else np.array([100])
    # TODO Si tu change les chiffres dans le logspace, le dernier chiffre va être (top value) - (min value) + 1.
    #  Par exemple, 5 - (-2) + 1 = 8
    # C'est pour avoir des multiples de 1, e.g. 1e-2, 1e-1, 1, 10, 100...
    results = np.zeros((test_encoded.shape[0], regs.shape[0]))
    index = 0
    error = []
    if family == "binomial":
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

            probs = fam_clm.predict(test_encoded)
            preds_01 = probs > np.mean(train["clm"]) # TODO : a revoir. cutoff = proportion de claims en entraînement

            fam_gamma = FairnessAwareModel(regularization=reg, protected_values=protected_values, family="gamma",
                                           alpha=(0.3429485))
            fam_gamma.fit(train_encoded, reg_claim_train, warm_start=True)

            results[:, index] = (fam_gamma.predict(test_encoded) * fam_clm.predict(test_encoded)).reshape(-1,)
            index += 1

    new_colnames = regs.tolist()
    new_colnames.insert(0, "which_set")
    df_to_return = pd.concat([pd.DataFrame(test["which_set"]).reset_index(drop=True), pd.DataFrame(results).reset_index(drop=True)],axis=1)
    df_to_return.columns = new_colnames

    path = join(dirname(abspath(__file__)), f"results_crossval_{family}_logspace-2_5_50.csv")
    df_to_return.to_csv(path, index=False)




