library(tidyverse);library(MASS);library(fairness);library(latex2exp)

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
# discrimination directe
direct = glm(clm~veh_value+veh_body+veh_age+area+gender, family=binomial(link="logit"), data=train)
summary(direct)

# discrimination indirecte
indirect = glm(clm~veh_value+veh_body+veh_age+area, family=binomial(link="logit"), data=train)

test$probs_direct = predict(direct, newdata=test, type="response")
test$probs_indirect = predict(indirect, newdata=test, type="response")
cutoff = mean(train$clm)

#### GRAPHIQUES POUR LE SEXE

################################################################################
# Égalité des chances
# P(\hat{Y}=1 | Y=1,A=a) = P(\hat{Y}=1 | Y=1,A=B) pour tout a,b
################################################################################
# égalité des chances (equal_odds dans la librairie)
mean_F = mean(test$probs_logit[test$gender=="F"])
mean_M = mean(test$probs_logit[test$gender=="M"])
mean(test$probs_logit)

# Y=1
equal_odds_1_direct = equal_odds(
  data    = test, 
  outcome = 'clm',
  outcome_base = '0',
  group   = 'gender',
  base    = 'F',
  probs   = 'probs_direct', 
  cutoff  = cutoff)

equal_odds_1_direct$Metric_plot + ggtitle("Égalité des chances avec la régression logistique")

# Y=1
equal_odds_1_indirect = equal_odds(
  data    = test, 
  outcome = 'clm',
  outcome_base = '0',
  group   = 'gender',
  base    = 'F',
  probs   = 'probs_indirect',
  cutoff  = cutoff)

equal_odds_1_indirect$Metric_plot + ggtitle("Égalité des chances avec la régression logistique")


# Y=0
equal_odds_0_direct = equal_odds(
  data    = test, 
  outcome = 'clm',
  outcome_base = '0',
  group   = 'gender',
  base    = 'F',
  probs   = 1-test$probs_direct, 
  cutoff  = 1-cutoff)

equal_odds_0_direct$Metric_plot + ggtitle("Égalité des chances avec la régression logistique")

# Y=0
equal_odds_0_indirect = equal_odds(
  data    = test, 
  outcome = 'clm',
  outcome_base = '0',
  group   = 'gender',
  base    = 'F',
  probs   = 1-test$probs_indirect,
  cutoff  = 1-cutoff)

equal_odds_0_indirect$Metric_plot + ggtitle("Égalité des chances avec la régression logistique")


### Créer graphiques
max_ratio = function(fairness_output){
  ratio = fairness_output$Metric[1,1]/fairness_output$Metric[1,2]
  if(ratio > 1){
    ratio = 1/ratio
  }
  # ratio est entre 0 et 1, ne doit pas être inférieur à 0.8 pour éviter un 
  # "effet disproportionné".
  # Philippe Besse. Détecter, évaluer les risques des impacts discriminatoires des algorithmes d'IA. 2020. hal-02616963
  return(ratio)
}

max_ratio(equal_odds_0_direct)

object_name_as_str = function(objet){
    deparse(substitute(objet))
}


object_name_as_str(equal_odds_0_indirect)

create_df = function(...){
  #' ... est un ensemble d'objets retournées par fairness
  #' Retourne un dataset qui sera utilisé pour les graphiques
  
  df = data.frame()
  object_names=sapply(substitute(list(...))[-1], deparse)
  
  arguments <- list(...)
  # création du dataset pour chaque modèle testé
  i=1
  for(objet in arguments){
    name = object_names[i]

    if (stringr::str_detect(name, "equal_odds")){
      metric = "Equalized odds"
    }
    
    if (stringr::str_detect(name, "1")){
      metric = paste0(metric, ", Y=1")
    } else{
      metric = paste0(metric, ", Y=0")
    }
    
    if (stringr::str_detect(name, "indirect")){
      model = "Indirecte"
    } else if (stringr::str_detect(name, "penalized")){
      model = "Pénalité"
    } else{
      model = "Directe"
    }
    
    df = rbind(df, c(model, metric, max_ratio(objet)))
    i = i+1
  }
  colnames(df) = c("Modèle", "Métrique", "Ratio.max")
  df$Ratio.max = as.numeric(df$Ratio.max)
  
  return(df)
}

create_equity_plot = function(dataset, title){
  scaleFUN <- function(x) sprintf("%.1f", x) # round y tickslabel to 1 decimal
  
  ggplot(dataset,
         aes(x = `Modèle`,
             y = `Ratio.max`,
             fill = `Métrique`)) +
    geom_bar(stat = "identity", # y is actual bar height
             position = position_dodge()) + # unstack bars
    labs(y="Ratio maximal", title=TeX(title)) +
    theme_bw() + 
    geom_hline(yintercept=1.25, col="red") +
    scale_y_continuous(labels=scaleFUN)
      
}

df = create_df(equal_odds_1_direct, equal_odds_1_indirect, equal_odds_0_direct, equal_odds_0_indirect)
create_equity_plot(df, "Equalized odds du genre selon le modèle utilisé")










### TODO : Finir la fonction pour générer automatiquement le dataset

### TODO : préparer les graphiques pour la loi gamma en python probablement











































#######
# OLD
########
################################################################################
# Parité démographique
# P(\hat{Y}=1 | S=s) = P(\hat{Y}=1 |S!=s) pour tout y, A et b
################################################################################
library(fairness)

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

################################################################################
# Parité démographique
# P(\hat{Y}=1 | S=s) = P(\hat{Y}=1 |S!=s) pour tout y, A et b
################################################################################
dem_parity_logit = dem_parity(
  data    = test, 
  outcome = 'clm',
  outcome_base = '0',
  group   = 'agecat',
  base    = '1',
  probs   = 'probs_logit', 
  cutoff  = mean(test$probs_logit)) # TODO : comment choisir un bon cutoff? prendre la moyenne est le genre de cutoff qu'on prend en analyse discriminante (linéaire) à deux classes...

dem_parity_xgb = equal_odds(
  data    = test, 
  outcome = 'clm',
  outcome_base = '0',
  group   = 'agecat',
  base    = '1',
  probs   = 'probs_xgb', 
  cutoff  = mean(test$probs_xgb))

dem_parity_xgb$Metric_plot + ggtitle("Égalité des chances avec XGBoost")
dem_parity_logit$Metric_plot + ggtitle("Égalité des chances avec la régression logistique") # inéquitable

dem_parity_logit

################################################################################
# Égalité de l'exactitude
# P(\hat{Y}=y | S=s) = P(\hat{Y}=y |S!=s) pour tout y, A et b
################################################################################
# Tremblay, 2021 dit qu'il est préférable d'utiliser le taux de faux positifs et de faux négatifs en complément

acc_parity_logit = acc_parity(
  data    = test, 
  outcome = 'clm',
  outcome_base = '0',
  group   = 'agecat',
  base    = '1',
  probs   = 'probs_logit', 
  cutoff  = mean(test$probs_logit)) # TODO : comment choisir un bon cutoff?

fnr_parity_logit = fnr_parity(
  data    = test, 
  outcome = 'clm',
  outcome_base = '0',
  group   = 'agecat',
  base    = '1',
  probs   = 'probs_logit', 
  cutoff  = mean(test$probs_logit)) # TODO : comment choisir un bon cutoff?

fpr_parity_logit = fpr_parity(
  data    = test, 
  outcome = 'clm',
  outcome_base = '0',
  group   = 'agecat',
  base    = '1',
  probs   = 'probs_logit', 
  cutoff  = mean(test$probs_logit)) # TODO : comment choisir un bon cutoff?


acc_parity_logit
fnr_parity_logit
fpr_parity_logit

################################################################################
# Égalité des chances
# P(\hat{Y}=1 | Y=1,A=a) = P(\hat{Y}=1 | Y=1,A=B) pour tout a,b
################################################################################

equal_odds_logit = equal_odds(
  data    = test, 
  outcome = 'clm',
  outcome_base = '0',
  group   = 'agecat',
  base    = '1',
  probs   = 'probs_logit', 
  cutoff  = mean(test$probs_logit)) # TODO : comment choisir un bon cutoff?

equal_odds_logit
equal_odds_logit$Metric_plot + ggtitle("Égalité des chances avec la régression logistique")

################################################################################
# Égalité de la précision
# P(Y=1 | \hat{Y}=1,A=a) = P(Y=1 | \hat{Y}=1,A=b) pour tout A et b
################################################################################
pred_rate_logit = pred_rate_parity(
  data    = test, 
  outcome = 'clm',
  outcome_base = '0',
  group   = 'agecat',
  base    = '1',
  probs   = 'probs_logit', 
  cutoff  = mean(test$probs_logit)) # TODO : comment choisir un bon cutoff?

pred_rate_logit
