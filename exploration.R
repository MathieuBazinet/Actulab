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

test$probs_xgb = predict(xgb, test_matrix)

hist(test$probs_xgb)
hist(test$probs_logit)

################################################################################
# Équité dans le modèle de Poisson
################################################################################
library(fairness)

# égalité des chances (equal_odds dans la librairie)
mean_F = mean(test$probability[test$gender=="F"])
mean_M = mean(test$probability[test$gender=="M"])
mean(test$probability)

equal_odds_logit = equal_odds(
  data    = test, 
  outcome = 'clm',
  outcome_base = '0',
  group   = 'gender',
  base    = 'F',
  probs   = 'probability', 
  cutoff  = mean(test$probs_logit)) # TODO : comment choisir un bon cutoff?

equal_odds_xgb = equal_odds(
  data    = test, 
  outcome = 'clm',
  outcome_base = '0',
  group   = 'gender',
  base    = 'F',
  probs   = 'probability', 
  cutoff  = mean(test$probs_xgb))

equal_odds_xgb$Metric_plot
equal_odds_logit$Metric_plot
