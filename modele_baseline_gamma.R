library(tidyverse);library(MASS);library(fairness)

df = read.csv("dataCar.csv")
df = df %>%
  dplyr::select(-c(X_OBSTAT_)) %>%
  mutate(
    veh_body = as.factor(veh_body),
    veh_age = as.factor(veh_age),
    gender = as.factor(gender),
    area = as.factor(area),
    agecat = as.factor(agecat)
  )

################################################################################
# Train-test spit et exporter les données pour Python
################################################################################

library(caTools)
set.seed(42)
sample = sample.split(df$clm, SplitRatio = .8)
train = subset(df, sample == TRUE)
test  = subset(df, sample == FALSE)

df$train = sample
#write.csv(df, "dataCar_clean.csv",row.names=FALSE)

################################################################################
# Régression gamma
################################################################################
# juste pour voir le genre de sortie et les paramètres estimés par la fonction
df_claimsg0 = subset(train, subset=claimcst0>0)

gam = glm(claimcst0 ~ factor(agecat) + gender + area, data=df_claimsg0, family = Gamma(link="log"))
summary(gam)

gam$coefficients
gam$effects
gam$method

### Estimation du paramètre de dispersion (shape, alpha=nu)
### remarque : dispersion = phi = nu^-1
# https://stats.stackexchange.com/questions/367560/how-to-choose-shape-parameter-for-gamma-distribution-when-using-generalized-line
# https://courses.ms.ut.ee/MTMS.01.011/2019_spring/uploads/Main/GLM_slides_5_continuous_response_print.pdf

summary(gam)$disp # estimation de alpha=nu=2.893
MASS::gamma.shape(gam) # 0.7632
# pas du tout la même chose...

?MASS:::gamma.shape.glm

#Dispersion parameter estimation
#1.Deviance method
(gam$deviance/gam$df.residual)  # 1.569
#2.Pearson method
sum(resid(gam,type='pear')^2)/gam$df.residual # estimation used in summary(model)=2.893


################################################################################
# Analyse de la distribution des montants réclamés à l'intérieur des quantiles
# on ne veut pas avoir slm des hommes dans le 3e-4e quantile, p.ex.
################################################################################






################################################################################
# Fonctions pour les graphiques
################################################################################
RMSE = function(true, prediction){
  sqrt(sum((true-prediction)^2))
}
# peut-être utiliser une normalisation pour que ce soit plus interprétable

# les quantiles sont calculées sur les données d'entraînement. En espérant que les données d'entraînement représentent bien les données de validation et de test...
quantiles_train = quantiles(train$numclaims, probs=)
WAGF = function(train, valid, predicted_outcome, true_risk, protected_attribute, protected_attribute_base, quantiles=c(0,0.25,0.5,0.75,1)){
  #' Calculer Weak Actulab Group Fairness définie comme E(M|Q_{\alpha_i} <= R <= Q_{\alpha_{i+1}}, A=a) - E(M|Q_{\alpha_i} <= R <= Q_{\alpha_{i+1}}, A!=a) = \delta. R=true risk=montant réclamé, M=montant réclamé PRÉDIT, A=attribut protégé, e.g. gender.
  #' On calcule la différence de l'espérance pour tous les quantiles et on retourne la moyenne, le max et la médiane.
  #' Suppose que l'attribut protégé a deux modalités (homme/femme) pour simplifier
  
  quantiles_train = quantile(train[,true_risk], probs=quantiles)
  
  df_1 = valid[valid[,protected_attribute]==protected_attribute_base & valid[,predicted_outcome] %in% quantiles_train]
  df_2 = valid[valid[,protected_attribute]!=protected_attribute_base & valid[,predicted_outcome] %in% quantiles_train]
  
  mean(df1[,predicted_outcome]) - mean(df2[,predicted_outcome])
  
}

















