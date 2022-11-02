import numpy as np
import pandas as pd

dataCar = pd.read_csv("./dataCar.csv")
dataCar = dataCar.drop("X_OBSTAT_", axis=1)
# TODO encoder correctement les différentes valeurs qui ne sont pas des entiers
# dataCar.replace(to_replace=np.unique(dataCar['gender']), inplace=True, value=range(np.unique(dataCar['gender']).shape[0]))
# dataCar.replace(to_replace=np.unique(dataCar['veh_body']), inplace=True, value=range(np.unique(dataCar['veh_body']).shape[0]))
# dataCar.replace(to_replace=np.unique(dataCar['area']), inplace=True, value=range(np.unique(dataCar['area']).shape[0]))
