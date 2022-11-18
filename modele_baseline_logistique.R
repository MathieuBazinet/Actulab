library(tidyverse);library(MASS);library(caret);library(fairness);library(latex2exp);library(pROC)

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
# 
# library(caTools)
# set.seed(42)
# sample_train = sample.split(df$clm, SplitRatio = .7)
# train = subset(df, sample_train == TRUE)
# test  = subset(df, sample_train == FALSE)


ids_split <- splitTools::partition(
  y = df[, "clm"]
  ,p = c(train = 0.7, valid = 0.15, test = 0.15)
  ,type = "stratified" # stratified is the default
  ,seed = 42
)
train <- df[ids_split$train, ]
valid <- df[ids_split$valid, ]
test <- df[ids_split$test, ]

train$which_set = 0
valid$which_set = 1
test$which_set = 2

#df = rbind(train,valid,test)
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

# Y=1
equal_odds_1_direct = equal_odds(
  data    = test, 
  outcome = 'clm',
  #outcome_base = '0',
  group   = 'agecat',
  base    = '1',
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
equal_odds_0_direct = fpr_parity(
  data    = test, 
  outcome = 'clm',
  outcome_base = '0',
  group   = 'gender',
  base    = 'F',
  probs   = test$probs_direct, 
  cutoff  = cutoff)

equal_odds_0_direct$Metric_plot + ggtitle("Égalité des chances avec la régression logistique")

# Y=0
equal_odds_0_indirect = fpr_parity(
  data    = test, 
  outcome = 'clm',
  outcome_base = '0',
  group   = 'gender',
  base    = 'F',
  probs   = test$probs_indirect,
  cutoff  = cutoff)

equal_odds_0_indirect$Metric_plot + ggtitle("Égalité des chances avec la régression logistique")






### Créer graphiques
min_ratio = function(fairness_output){
  ratio = fairness_output$Metric[1,1]/fairness_output$Metric[1,2]
  #if(ratio > 1){
  #  ratio = 1/ratio
  #}
  # ratio est entre 0 et 1, ne doit pas être inférieur à 0.8 pour éviter un 
  # "effet disproportionné".
  # Philippe Besse. Détecter, évaluer les risques des impacts discriminatoires des algorithmes d'IA. 2020. hal-02616963
  return(ratio)
}

min_ratio(equal_odds_0_direct)

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
    
    df = rbind(df, c(model, metric, min_ratio(objet)))
    i = i+1
  }
  colnames(df) = c("Modèle", "Métrique", "Ratio.min")
  df$Ratio.min = as.numeric(df$Ratio.min)
  
  return(df)
}

create_equity_plot = function(dataset, title, position="dodge"){
  #'@param position (default="dodge"). Valeur de l'argument "position" pour `geom_bar`. En praticulier, "stack" pour stacker les barres ou "dodge" pour les mettre les unes à côté des autres.
  scaleFUN <- function(x) sprintf("%.1f", x) # round y tickslabel to 1 decimal
  
  p <- ggplot(dataset,
         aes(x = `Modèle`,
             y = `Ratio.min`,
             fill = `Métrique`)) +
    geom_bar(stat = "identity", # y is actual bar height
             position = position) + # unstack bars
    labs(y="Ratio proba.", title=TeX(title)) +
    theme_bw() + 
    scale_y_continuous(labels=scaleFUN)
  
  if (position=="dodge"){
    p = p + geom_hline(yintercept=0.8, col="red") + geom_hline(yintercept=1.25, col="red") + geom_hline(yintercept=1, col="black", linetype="dashed")
  }  
  return(p)
}

new_df = create_df(equal_odds_1_direct, equal_odds_1_indirect, equal_odds_0_direct, equal_odds_0_indirect)
create_equity_plot(new_df, "Equalized odds du genre selon le modèle utilisé", position="dodge")




# fonction create df avec courbe ROC
# fonction graphiques avec sensibilité issue de la courbe ROC
roc_direct = pROC::roc(response=test$clm, test$probs_direct)
coords(roc_direct, x=cutoff)



# TODO : rouler le modèle fairness
# TODO : comment choisir un lambda? Maximiser quelle métrique? ROC/AUC OU sensibilité à un seuil donné

################################################################################
# Analyse des modèles pénalisés
################################################################################
# Rouler le fichier "main.py"

valid_test_probs = read.csv(file="results_crossval_logistic.csv")

valid_probs = subset(valid_test_probs, which_set==1) # validation
valid_probs$clm = valid$clm
valid_probs$gender = valid$gender

columns_with_probs = colnames(valid_probs) %>% str_detect("^X")
lambdas = colnames(valid_probs) %>% str_extract("\\d+.\\d+")

penalized_models = data.frame()

# calcul ROC/sensibilité/equalized odds pour voir l'effet de l'importance
# de la discrimination sur les performances
for (i in which(columns_with_probs)){
  roc_i = pROC::roc(response=valid$clm, valid_probs[,i])
  sens_i = coords(roc_i, x=cutoff)$sensitivity
  auc_i = pROC::auc(roc_i) %>% as.numeric()
  
  # Y=1
  equal_odds_1_i = equal_odds(
    data    = valid_probs,
    outcome = 'clm',
    outcome_base = '0',
    group   = 'gender',
    base    = 'F',
    probs   = valid_probs[,i],
    cutoff  = cutoff) %>% 
    min_ratio()

  # Y=0
  equal_odds_0_i = fpr_parity(
    data    = valid_probs,
    outcome = 'clm',
    outcome_base = '0',
    group   = 'gender',
    base    = 'F',
    probs   = valid_probs[,i],
    cutoff  = cutoff) %>%
    min_ratio()
  
  values_i = c(auc_i, sens_i, equal_odds_1_i, equal_odds_0_i, as.numeric(lambdas[i]))
  print(lambdas[i])
  penalized_models = rbind(penalized_models, values_i)
  colnames(penalized_models) = c("AUC", "Sensibilité", "Equalized odds, Y=1", "Equalized odds, Y=0", "lambda")
}

library(scales)
show_col(hue_pal()(4))


# Effet de lambda sur l'équité
colors <- c("Equalized odds, Y=0" = hue_pal()(2)[1], "Equalized odds, Y=1" = hue_pal()(2)[2])
p1 = ggplot(penalized_models) + 
  geom_line(aes(x=lambda, y=`Equalized odds, Y=0`, color="Equalized odds, Y=0")) + 
  geom_line(aes(x=lambda, y=`Equalized odds, Y=1`, color="Equalized odds, Y=1")) +
  scale_color_manual(values = colors) +
  geom_hline(yintercept=0.8, col="red") + 
  geom_hline(yintercept=1.25, col="red") + 
  geom_hline(yintercept=1, col="black", linetype="dashed") +
  labs(title="Effet de la pénalité sur l'équité", x=TeX("$\\lambda$"), y="Ratio des probabilités hommes/femmes", color="Métrique") +
  theme_bw()


# Effet de lambda sur les performances
colors <- c("AUC" = hue_pal()(2)[1], "Sensibilité" = hue_pal()(2)[2])
p2 = ggplot(penalized_models) + 
  geom_line(aes(x=lambda, y=AUC, color="AUC")) +
  geom_line(aes(x=lambda, y=Sensibilité, color="Sensibilité")) +
  labs(title="Effet de la pénalité sur la sensibilité (cutoff=0.068) et l'AUC") +
  scale_color_manual(values = colors) +
  labs(title="Effet de la pénalité sur la performance", x=TeX("$\\lambda$"), y="Métrique", color="Métrique") +
  theme_bw()

library(patchwork)
p1/p2


# TODO : diagrammes à barre avec le modèle choisi (mettre une ligne vertical pour le modèle choisi sur les time series)
# TODO : refaire l'analyse avec parité démographique






























































# equal_odds_1 = function(dataset, outcome, outcome_base, probabilities, cutoff, protected_attribute, protected_attribute_base){
#   s1 = dataset[dataset[,protected_attribute]==protected_attribute_base,]
#   s2 = dataset[dataset[,protected_attribute]!=protected_attribute_base,]
#   
#   pred_1 = as.integer(s1[,probabilities] > cutoff)
#   pred_2 = as.integer(s2[,probabilities] > cutoff)
#   
#   table(a=pred_1, b=s1[,outcome]) %>% print()
#   sensitivity_1 = caret::sensitivity(data=as.factor(pred_1),reference=as.factor(s1[,outcome]), positive="1")
#   sensitivity_2 = caret::sensitivity(data=as.factor(pred_2),reference=as.factor(s2[,outcome]), positive="1")
#   
#   print(sensitivity_1)
#   print(sensitivity_2)
#   
#   return(sensitivity_2/sensitivity_1)
# }
# equal_odds_1(test, outcome="clm", outcome_base=1, probabilities="probs_direct", cutoff=cutoff, protected_attribute="gender", protected_attribute_base="F")