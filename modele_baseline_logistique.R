library(tidyverse);library(MASS);library(caret);library(fairness);library(latex2exp);library(pROC);library(scales);library(patchwork)

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

df_claimsg0 <- subset(df, subset=claimcst0>0) # df sans les motants nuls 

subset(df, subset=claimcst0>0 & claimcst0<=15000) %>% # il y a 65 obs avec claimcst0>15000 
  ggplot() +
  geom_histogram(aes(x=claimcst0)) +
  theme_bw() +
  xlab("Montant de la réclamation") +
  ylab("Fréquence")

subset(df, subset=claimcst0>0 & claimcst0<=15000) %>% # il y a 65 obs avec claimcst0>15000 
  ggplot() +
  geom_histogram(aes(x=claimcst0, fill=gender, y=stat(count / sum(count)), alpha=0.3)) +
  theme_bw() +
  xlab("Montant de la réclamation") +
  ylab("Fréquence")

library('lattice')
sub = subset(df, subset=claimcst0>0 & claimcst0<=15000)
histogram(~ claimcst0 | gender, data = sub) # se ressemble bcp entre les 2 sexes,
# les différences doivent être dans les valeurs extrêmes?

table(df$clm, df$gender) %>% prop.table(margin=2)
mean(df$clm)

summary(df$claimcst0)
summary(df_claimsg0$claimcst0)

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

valid$probs_direct = predict(direct, newdata=valid, type="response")
valid$probs_indirect = predict(indirect, newdata=valid, type="response")
test$probs_direct = predict(direct, newdata=test, type="response")
test$probs_indirect = predict(indirect, newdata=test, type="response")
cutoff = mean(train$clm)

################################################################################
# Fonctions utiles
################################################################################
### Créer graphiques
min_ratio = function(fairness_output){
  ratio = fairness_output$Metric[1,2]/fairness_output$Metric[1,1] # homme/femme
  #if(ratio > 1){
  #  ratio = 1/ratio
  #}
  # ratio est entre 0 et 1, ne doit pas être inférieur à 0.8 pour éviter un 
  # "effet disproportionné".
  # Philippe Besse. Détecter, évaluer les risques des impacts discriminatoires des algorithmes d'IA. 2020. hal-02616963
  return(ratio)
}

#min_ratio(equal_odds_0_direct)

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



################################################################################
# Graphiques
################################################################################

new_df = create_df(equal_odds_1_direct, equal_odds_1_indirect, equal_odds_0_direct, equal_odds_0_indirect)
create_equity_plot(new_df, "Equalized odds du genre selon le modèle utilisé", position="dodge")






















################################################################################
# Analyse des modèles pénalisés
################################################################################
# Rouler le fichier "main.py"

valid_test_probs = read.csv(file="./resultats/results_crossval_binomial_linspace_-2_4_20_EO.csv") # résultats assez étranges avec results_crossval_binomial_logspace_-2_5_50.csv

valid_probs = subset(valid_test_probs, which_set==1) # validation
valid_probs$clm = valid$clm
valid_probs$gender = valid$gender

columns_with_probs = colnames(valid_probs) %>% str_detect("^X")
lambdas = colnames(valid_probs) %>% str_extract("\\d+.\\d+")

penalized_models = data.frame()

# calcul ROC/sensibilité/equalized odds pour voir l'effet de l'importance
# de la discrimination sur les performances
# DONNÉES DE VALIDATION
for (i in which(columns_with_probs)){
  roc_i = pROC::roc(response=valid_probs$clm, valid_probs[,i])
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
  
  values_i = c(auc_i, sens_i, equal_odds_1_i, equal_odds_0_i, as.numeric(lambdas[i]), 1, 5/4, 4/5)
  print(lambdas[i])
  penalized_models = rbind(penalized_models, values_i)
  colnames(penalized_models) = c("AUC", "Sensibilité", "Equalized odds, Y=1", "Equalized odds, Y=0", "lambda", "optimal=1", "Ratio 4/5", "Ratio 5/4")
}


# Effet de lambda sur l'équité
# Y=1
equal_odds_1_direct_valid = equal_odds(
  data    = valid, 
  outcome = 'clm',
  #outcome_base = '0',
  group   = 'gender',
  base    = 'F',
  probs   = 'probs_direct', 
  cutoff  = cutoff)

# Y=1
equal_odds_1_indirect_valid = equal_odds(
  data    = valid, 
  outcome = 'clm',
  outcome_base = '0',
  group   = 'gender',
  base    = 'F',
  probs   = 'probs_indirect',
  cutoff  = cutoff)

# Y=0
equal_odds_0_direct_valid = fpr_parity(
  data    = valid, 
  outcome = 'clm',
  outcome_base = '0',
  group   = 'gender',
  base    = 'F',
  probs   = 'probs_direct', 
  cutoff  = cutoff)
# Y=0
equal_odds_0_indirect_valid = fpr_parity(
  data    = valid, 
  outcome = 'clm',
  outcome_base = '0',
  group   = 'gender',
  base    = 'F',
  probs   = 'probs_indirect', 
  cutoff  = cutoff)

roc_direct_valid = pROC::roc(response=valid_probs$clm, valid$probs_direct)
sens_direct_valid = coords(roc_direct_valid, x=cutoff)$sensitivity
auc_direct_valid = pROC::auc(roc_direct_valid) %>% as.numeric()
roc_indirect_valid = pROC::roc(response=valid_probs$clm, valid$probs_indirect)
sens_indirect_valid = coords(roc_indirect_valid, x=cutoff)$sensitivity
auc_indirect_valid = pROC::auc(roc_indirect_valid) %>% as.numeric()

colors <- c("Equalized odds, Y=0" = hue_pal()(2)[1], "Equalized odds, Y=1" = hue_pal()(2)[2], "Ratio optimal" = "black", "Ratio 4/5 ou 5/4"="red")
p1 = ggplot(penalized_models) + 
  geom_line(aes(x=lambda, y=`Equalized odds, Y=0`, color="Equalized odds, Y=0")) + 
  geom_point(aes(x=lambda, y=`Equalized odds, Y=0`, color="Equalized odds, Y=0")) +
  geom_line(aes(x=lambda, y=`Equalized odds, Y=1`, color="Equalized odds, Y=1")) +
  geom_point(aes(x=lambda, y=`Equalized odds, Y=1`, color="Equalized odds, Y=1")) +
  geom_line(aes(x=lambda, y=`optimal=1`, color="Ratio optimal")) +
  geom_line(aes(x=lambda, y=`Ratio 4/5`, color="Ratio 4/5 ou 5/4")) +
  geom_line(aes(x=lambda, y=`Ratio 5/4`, color="Ratio 4/5 ou 5/4")) +
  scale_color_manual(values = colors) +
  geom_point(aes(x=0, y=min_ratio(equal_odds_1_direct_valid)), colour=colors[2],shape=3, size=3) +
  geom_point(aes(x=0, y=min_ratio(equal_odds_0_direct_valid)), colour=colors[1], shape=3, size=3) +
  geom_point(aes(x=0, y=min_ratio(equal_odds_1_indirect_valid)), colour=colors[2], shape=4, size=3) +
  geom_point(aes(x=0, y=min_ratio(equal_odds_0_indirect_valid)), colour=colors[1], shape=4, size=3) +
  labs(title="Effet de la pénalité sur l'équité", x=TeX("$\\lambda$"), y="Ratio des probabilités hommes/femmes", color="Métrique") +
  theme_bw()
# "+" = directe, "x" = indirecte. Le mettre en notes de bas de page dans le beamer car je n'ai pas trouvé comment l'ajouter dans le graphique.
p1


# Effet de lambda sur les performances
colors <- c("AUC" = hue_pal()(2)[1], "Sensibilité" = hue_pal()(2)[2])
p2 = ggplot(penalized_models) + 
  geom_line(aes(x=lambda, y=AUC, color="AUC")) +
  geom_point(aes(x=lambda, y=AUC, color="AUC")) +
  geom_line(aes(x=lambda, y=Sensibilité, color="Sensibilité")) +
  geom_point(aes(x=lambda, y=Sensibilité, color="Sensibilité")) +
  scale_color_manual(values = colors) +
  geom_point(aes(x=0, y=auc_direct_valid), colour=colors[1],shape=3, size=3) +
  geom_point(aes(x=0, y=sens_direct_valid), colour=colors[2],shape=3, size=3) +
  geom_point(aes(x=0, y=auc_indirect_valid), colour=colors[1], shape=4, size=3) +
  geom_point(aes(x=0, y=sens_indirect_valid), colour=colors[2], shape=4, size=3) +
  labs(title="Effet de la pénalité sur la performance", x=TeX("$\\lambda$"), y="Métrique", color="Métrique") +
  theme_bw()
# "+" = directe, "x" = indirecte. Le mettre en notes de bas de page dans le beamer car je n'ai pas trouvé comment l'ajouter dans le graphique.
p2

p3 = p1/p2
#ggsave(file="./figures/performance_equite_binomial.pdf", plot=p3, width=10, height=8)

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