library(tidyverse);library(MASS)

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
# Statistiques descriptives
################################################################################

round(df$numclaims %>% table() %>% prop.table()*100,2) %>% cumsum()
# 99.58% des données ont 0 ou 1 réclamation. Dans ce contexte, il est approprié
# d'utiliser une variable binaire, c'est-à-dire considérer s'il y a au moins une
# réclamation dans l'année. Le GLM sera donc une régression logistique.
# Nous utiliserons comme variable y la variable `clm`.

################################################################################
# Attributs protégés choisis
################################################################################
# gender
# agecat

################################################################################
# Régression logistique
################################################################################
formule = formula(clm~veh_value+veh_body+veh_age+area)

library(caTools)
set.seed(42)
sample = sample.split(df$clm, SplitRatio = .8)
train = subset(df, sample == TRUE)
test  = subset(df, sample == FALSE)

logistique = glm(formule, family=binomial(link="logit"), data=train)
summary(logistique)

test$probs_logit = predict(logistique, newdata=test, type="response")

################################################################################
# XGBoost
################################################################################
library(xgboost)
library(Matrix)
train_matrix = sparse.model.matrix(~clm+veh_value+veh_body+veh_age+area, data=train) # sparse.model.matrix ou model.matrix

xgb <- xgboost(data = train_matrix, label = train$clm, max.depth = 2, eta = 1, nthread = 2, nrounds = 2, objective = "binary:logistic")

test_matrix = sparse.model.matrix(~clm+veh_value+veh_body+veh_age+area, data=test)

test$probs_xgb = predict(xgb, test_matrix, type="response")

preds_xgb = test$probs_xgb > 0.5

mean(test$clm==preds_xgb) # accuracy de 100%, le modèle XGBoost classifie parfaitement. Les métriques d'inéquité ne seront pas intéressantes pour ce modèle.

hist(test$probs_xgb) # très (trop?) bon pour classer?
hist(test$probs_logit)


#### GRAPHIQUES POUR LE SEXE

################################################################################
# Parité démographique
# P(\hat{Y}=1 | S=s) = P(\hat{Y}=1 |S!=s) pour tout y, A et b
################################################################################
dem_parity_logit = dem_parity(
  data    = test, 
  outcome = 'clm',
  outcome_base = '0',
  group   = 'gender',
  base    = 'F',
  probs   = 'probs_logit', 
  cutoff  = mean(test$probs_logit)) # TODO : comment choisir un bon cutoff?

dem_parity_xgb = equal_odds(
  data    = test, 
  outcome = 'clm',
  outcome_base = '0',
  group   = 'gender',
  base    = 'F',
  probs   = 'probs_xgb', 
  cutoff  = mean(test$probs_xgb))

dem_parity_xgb$Metric_plot + ggtitle("Égalité des chances avec XGBoost")
dem_parity_logit$Metric_plot + ggtitle("Égalité des chances avec la régression logistique") # inéquitable

################################################################################
# Égalité de l'exactitude
# P(\hat{Y}=y | S=s) = P(\hat{Y}=y |S!=s) pour tout y, A et b
################################################################################
# Tremblay, 2021 dit qu'il est préférable d'utiliser le taux de faux positifs et de faux négatifs en complément

acc_parity_logit = acc_parity(
  data    = test, 
  outcome = 'clm',
  outcome_base = '0',
  group   = 'gender',
  base    = 'F',
  probs   = 'probs_logit', 
  cutoff  = mean(test$probs_logit)) # TODO : comment choisir un bon cutoff?

fnr_parity_logit = fnr_parity(
  data    = test, 
  outcome = 'clm',
  outcome_base = '0',
  group   = 'gender',
  base    = 'F',
  probs   = 'probs_logit', 
  cutoff  = mean(test$probs_logit)) # TODO : comment choisir un bon cutoff?

fpr_parity_logit = fpr_parity(
  data    = test, 
  outcome = 'clm',
  outcome_base = '0',
  group   = 'gender',
  base    = 'F',
  probs   = 'probs_logit', 
  cutoff  = mean(test$probs_logit)) # TODO : comment choisir un bon cutoff?


acc_parity_logit
fnr_parity_logit
fpr_parity_logit

################################################################################
# Égalité des chances
# P(\hat{Y}=1 | Y=1,A=a) = P(\hat{Y}=1 | Y=1,A=B) pour tout a,b
################################################################################
library(fairness)

# égalité des chances (equal_odds dans la librairie)
mean_F = mean(test$probs_logit[test$gender=="F"])
mean_M = mean(test$probs_logit[test$gender=="M"])
mean(test$probs_logit)

equal_odds_logit = equal_odds(
  data    = test, 
  outcome = 'clm',
  outcome_base = '0',
  group   = 'gender',
  base    = 'F',
  probs   = 'probs_logit', 
  cutoff  = mean(test$probs_logit)) # TODO : comment choisir un bon cutoff?

equal_odds_logit$Metric_plot + ggtitle("Égalité des chances avec la régression logistique")

################################################################################
# Égalité de la précision
# P(Y=1 | \hat{Y}=1,A=a) = P(Y=1 | \hat{Y}=1,A=b) pour tout A et b
################################################################################
pred_rate_logit = pred_rate_parity(
  data    = test, 
  outcome = 'clm',
  outcome_base = '0',
  group   = 'gender',
  base    = 'F',
  probs   = 'probs_logit', 
  cutoff  = mean(test$probs_logit)) # TODO : comment choisir un bon cutoff?

pred_rate_logit


#### GRAPHIQUES POUR LA CATÉGORIE D'ÂGE
