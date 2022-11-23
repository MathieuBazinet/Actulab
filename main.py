import numpy as np
import pandas as pd
from Fairness_aware_model import FairnessAwareModel
from os.path import join, dirname, abspath
try:
    sm_api_possible = True
    import statsmodels.api as sm
except:
    sm_api_possible = False

def hot_encoder(data, optional_columns):
    """
    Fonction permettant d'encoder sous forme one-hot la base de données
    :param data: La base de données a encoder
    :param optional_columns: Des colonnes d'attributs protégés qui doivent obligatoirement être encodées.
    :return: La base de donnée encodée
    """
    categorical_cols = []
    for column in data.columns:
            if data[column].dtype == object or column in optional_columns:
                categorical_cols.append(column)

    to_return = pd.get_dummies(data, columns = categorical_cols)
    return to_return


if __name__ == "__main__":
    # Pour l'instant, vous pouvez choisir entre une "poisson", une "binomial" et une "gamma"
    family = "gamma"
    # Il serait par exemple possible d'utiliser aussi 'age_cat'
    protected_attributes = ['gender']

    # Lorsque cross_val est True, on entraîne sur multiples valeurs de lambda. Sinon, on utilise lambda=100
    cross_val = True

    # On fait le traitement nécessaire au fichier pour qu'il soit utilisé.
    dataCar = pd.read_csv("./dataCar_clean.csv")
    train = dataCar.loc[dataCar['which_set'] == 0]
    test = dataCar.loc[dataCar['which_set'] != 0] # Ceci est les données de validation ET de test

    protected_values = train[protected_attributes].values
    train = train.drop(protected_attributes, axis=1)
    test = test.drop(protected_attributes, axis=1)

    # retirer agecat du dataset au début pour voir ce qui se passe
    # avec gender "indépendamment" de la pénalisation sur agecat
    train = train.drop("agecat", axis=1)
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

    # ajout de l'ordonnée à l'origine 
    # REMARQUE : L'ajout d'une constante n'était pas utilisée pour la régression logistique dans nos résultats
    if sm_api_possible:
        train_encoded = sm.add_constant(train_encoded)
        test_encoded = sm.add_constant(test_encoded)

    #regs = np.logspace(-2, 4, 50) if cross_val else np.array([100])
    regs = np.linspace(0.01, 10000, 20) if cross_val else np.array([100]) # pour logistique

    # Fichiers de résultats de tests
    results = np.zeros((test_encoded.shape[0], regs.shape[0]))
    results_gamma = np.zeros((test_encoded.shape[0], regs.shape[0]))
    index = 0
    error = []
    if family == "binomial":
        for reg in regs:
            fam_logistic = FairnessAwareModel(regularization=reg, protected_values=protected_values, family="binomial",
                                              equity_metric="DP")
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
            fam_clm = FairnessAwareModel(regularization=reg, protected_values=protected_values, family="binomial",
                                         equity_metric="EO")
            fam_clm.fit(train_encoded, clm_train, warm_start=True)

            # alpha=phi^(-1) estimé en R avec summary(modele_gamma_discrimination)$disp = phi
            fam_gamma = FairnessAwareModel(regularization=reg, protected_values=protected_values, family="gamma",
                                           alpha=(1/2.92382))
            fam_gamma.fit(train_encoded, reg_claim_train, warm_start=True)

            results[:, index] = (fam_clm.predict(test_encoded)).reshape(-1,)
            results_gamma[:, index] = (fam_gamma.predict(test_encoded)).reshape(-1,)
            index += 1

    new_colnames = regs.tolist()
    new_colnames.insert(0, "which_set")

    # Sauvegarder les données dans des fichiers csv
    if family!="gamma":
        df_to_return = pd.concat([pd.DataFrame(test["which_set"]).reset_index(drop=True), pd.DataFrame(results).reset_index(drop=True)],axis=1)
        df_to_return.columns = new_colnames

        path = join(dirname(abspath(__file__)), f"resultats/results_crossval_{family}_linspace_-2_4_50_DP.csv")
        df_to_return.to_csv(path, index=False)
    else:
        # enregistrer résultats binomial et gamma dans deux CSV différents
        df_bin = pd.concat([pd.DataFrame(test["which_set"]).reset_index(drop=True), pd.DataFrame(results).reset_index(drop=True)],axis=1)
        df_bin.columns = new_colnames

        df_gam = pd.concat([pd.DataFrame(test["which_set"]).reset_index(drop=True), pd.DataFrame(results_gamma).reset_index(drop=True)],axis=1)
        df_gam.columns = new_colnames

        path_bin = join(dirname(abspath(__file__)), f"resultats/results_crossval_{family}_linspace_-2_4_20_binomial_EO.csv")
        path_gam = join(dirname(abspath(__file__)), f"resultats/results_crossval_{family}_linspace_-2_4_20_gamma_WAGF.csv")
        df_bin.to_csv(path_bin, index=False)
        df_gam.to_csv(path_gam, index=False)