library(tidyverse);library(MASS);library(fairness);library(ggplot2);library(patchwork);library(latex2exp);library(scales)

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

# library(caTools)
# set.seed(42)
# sample = sample.split(df$clm, SplitRatio = .8)
# train = subset(df, sample == TRUE)
# test  = subset(df, sample == FALSE)

#df$train = sample
#write.csv(df, "dataCar_clean.csv",row.names=FALSE)z

ids_split <- splitTools::partition(
    y = df[, "clm"]
    ,p = c(train = 0.7, valid = 0.15, test = 0.15)
    ,type = "stratified" # stratified is the default
    ,seed = 42
)
train_g0 <- df[ids_split$train, ]
valid_g0 <- df[ids_split$valid, ]
test_g0 <- df[ids_split$test, ]

train_g0 <- train_g0[train_g0$claimcst0>0, ]
valid_g0 <- valid_g0[valid_g0$claimcst0>0, ]
test_g0 <- test_g0[test_g0$claimcst0>0, ]

train_g0$which_set = 0
valid_g0$which_set = 1
test_g0$which_set = 2


# tronquer les montants trop élevés... va diminuer le biais pour tout le monde avec des petites valeurs de montant réclamé mais l'augmenter pour les montants réclamés élevés.
# je vais le changer seulement en train
max= quantile(train_g0$claimcst0, 0.9) # 18045.16$
train_g0 = train_g0 %>% 
  mutate(
    claimcst0_trunc = if_else(claimcst0>max, max, claimcst0)
  )

################################################################################
# Régression gamma directe et indirecte 
################################################################################
# Direct 
direct = glm(claimcst0 ~ veh_value+veh_body+veh_age+gender+area, data=train_g0, family = Gamma(link="log"))
summary(direct)

# Indirect

indirect = glm(claimcst0 ~ veh_value+veh_body+veh_age  + area, data=train_g0, family = Gamma(link="log"))
summary(indirect)

### Estimation du paramètre de dispersion (shape, alpha=nu)
### remarque : dispersion = phi = nu^-1
# https://stats.stackexchange.com/questions/367560/how-to-choose-shape-parameter-for-gamma-distribution-when-using-generalized-line
# https://courses.ms.ut.ee/MTMS.01.011/2019_spring/uploads/Main/GLM_slides_5_continuous_response_print.pdf

summary(indirect)$disp # estimation de alpha=nu=2.92382
MASS::gamma.shape(indirect) # 0.7632
# pas du tout la même chose...

?MASS:::gamma.shape.glm

#Dispersion parameter estimation
#1.Deviance method
(indirect$deviance/indirect$df.residual)  # 1.575598
#2.Pearson method
sum(resid(indirect,type='pear')^2)/indirect$df.residual # estimation used in summary(model)=2.92382



valid_g0$pred_direct <- predict( direct, newdata = valid_g0, type = "response")
valid_g0$pred_indirect <- predict( indirect, newdata = valid_g0, type = "response")

test_g0$pred_direct <-  predict( direct, newdata = test_g0, type = "response")
test_g0$pred_indirect <-predict( indirect, newdata = test_g0, type = "response")


################################################################################
# Analyse de la distribution des montants réclamés à l'intérieur des quantiles
# on ne veut pas avoir slm des hommes dans le 3e-4e quantile, p.ex.
################################################################################

# Excluant les 0, histogramme des montants réclamés
ggplot(train_g0, aes(x = claimcst0)) + 
  geom_histogram(colour = 1, fill = "white", bins=100) +
  #geom_density(col="red") + 
  labs(x="Montant réclamé", y="Densité", title="Montants réclamés (entraînement), excluant 0", fill="Genre") +
  theme_bw()
#ggsave()

# excluant les 0, histogramme des montants réclamés par sexe
g1 = ggplot(train_g0, aes(x=claimcst0, fill=gender)) + 
  geom_histogram(aes(y = ..density..),
                 colour = 1, bins=100, alpha=0.4) +
  #geom_density() + 
  labs(x="Montant réclamé", y="densité", fill="Genre") +
  theme_bw()


g2 = ggplot(train_g0, aes(x=claimcst0, fill=gender)) + 
  geom_density(aes(y = ..density..),
                 colour = 1, bins=100, alpha=0.4) +
  #geom_density() + 
  labs(x="Montant réclamé", y="densité", fill="Genre") +
  theme_bw()

g3 = ggpubr::ggarrange(g1, g2, ncol=2, nrow=1, common.legend = TRUE, legend="bottom") # les hommes ont des montants réclamés un peu plus élevés en pratique, queue un peu plus lourde...
# c'est vrai en entraînement mais pour toutes les données aussi
#ggsave("./figures/repartition_montant_reclamation_hf_train.pdf", plot=g3, width=10, height=6)


## pas très élaboré comme code mais ça fait la job 
## ech d'entrainement 
quantile_train_g0 <- quantile(train_g0$claimcst0)
train_q1 <- subset(train_g0, subset = (train_g0$claimcst0 < quantile_train_g0[[2]]) ) 
summary(train_q1)

train_q2 <- subset(train_g0, subset = ((train_g0$claimcst0 >= quantile_train_g0[[2]]) &
                                        (train_g0$claimcst0 < quantile_train_g0[[3]]))) 
summary(train_q2)

train_q3 <- subset(train_g0, subset = ((train_g0$claimcst0 >= quantile_train_g0[[3]]) &
                                        (train_g0$claimcst0 < quantile_train_g0[[4]]))) 
summary(train_q3)

train_q4 <- subset(train_g0, subset = ((train_g0$claimcst0 >= quantile_train_g0[[4]]))) 
summary(train_q4)

par(mfrow=c(2,2))
boxplot_q1 <-boxplot(train_q1$claimcst0~train_q1$gender, col=c("#009999", "#0000FF"), xlab = "Genre", ylab = "Montant réclamé", main="[min,Q1]")
boxplot_q2 <-boxplot(train_q2$claimcst0~train_q2$gender, col=c("#009999", "#0000FF"), xlab = "Genre", ylab = "Montant réclamé", main="[Q1,Q2]")
boxplot_q3 <-boxplot(train_q3$claimcst0~train_q3$gender, col=c("#009999", "#0000FF"), xlab = "Genre", ylab = "Montant réclamé", main="[Q3,Q4]")
boxplot_q4 <-boxplot(train_q4$claimcst0~train_q4$gender, col=c("#009999", "#0000FF"), xlab = "Genre", ylab = "Montant réclamé", main="[Q4,max]")
#mtext("Distribution du montant réclamé (excluant 0) dans les différents quantiles selon le genre", side = 3, line = - 1,font=2, outer = TRUE)
par(mfrow=c(1,1))

## ech de validation 
quantile_valid_g0 <- quantile(valid_g0$claimcst0)
valid_q1 <- subset(valid_g0, subset = (valid_g0$claimcst0 < quantile_valid_g0[[2]]) ) 
summary(valid_q1)

valid_q2 <- subset(valid_g0, subset = ((valid_g0$claimcst0 >= quantile_valid_g0[[2]]) &
                                        (valid_g0$claimcst0 < quantile_valid_g0[[3]]))) 
summary(valid_q2)

valid_q3 <- subset(valid_g0, subset = ((valid_g0$claimcst0 >= quantile_valid_g0[[3]]) &
                                        (valid_g0$claimcst0 < quantile_valid_g0[[4]]))) 
summary(valid_q3)

valid_q4 <- subset(valid_g0, subset = ((valid_g0$claimcst0 >= quantile_valid_g0[[4]]))) 
summary(valid_q4)

par(mfrow=c(2,2))
boxplot_q1 <-boxplot(valid_q1$claimcst0~valid_q1$gender, col=c("#009999", "#0000FF"), xlab = "Genre", ylab = "Montant réclamé", main="[min,Q1]")
boxplot_q2 <-boxplot(valid_q2$claimcst0~valid_q2$gender, col=c("#009999", "#0000FF"), xlab = "Genre", ylab = "Montant réclamé", main="[Q1,Q2]")
boxplot_q3 <-boxplot(valid_q3$claimcst0~valid_q3$gender, col=c("#009999", "#0000FF"), xlab = "Genre", ylab = "Montant réclamé", main="[Q3,Q4]")
boxplot_q4 <-boxplot(valid_q4$claimcst0~valid_q4$gender, col=c("#009999", "#0000FF"), xlab = "Genre", ylab = "Montant réclamé", main="[Q4,max]")
mtext("Distribution du montant réclamé (validation) dans les différents quantiles selon le genre", side = 3, line = - 1,font=2, outer = TRUE)
par(mfrow=c(1,1))





################################################################################
# Analyse de la distribution des montants réclamés prédits par les différents
# modèles
################################################################################


#Modèle direct
dat1 <- data.frame(dens = c(valid_g0$claimcst0, valid_g0$pred_direct)
                   , lines = rep(c("Montant reclamé", "Montant prédit"), each = nrow(valid_g0)))
#Plot.
ggplot(dat1, aes(x = dens, fill = lines)) + 
    geom_density(alpha = 0.5) +
    labs(y="Densité",x="Montant") +
  theme_bw()
#ggsave("./figures/distribution_clmcst0_direct_excluant0.png", width=8, height=6)

#Modèle indirect
dat2 <- data.frame(dens = c(valid_g0$claimcst0, valid_g0$pred_indirect)
                   , lines = rep(c("Montant reclamé", "Montant prédit"), each = nrow(valid_g0)))
#Plot.
ggplot(dat2, aes(x = dens, fill = lines)) + 
    geom_density(alpha = 0.5) +
    labs(y="Densité",x="Montant", title="Distribution des prédictions : Modèle indirect (excluant 0)")

### Séparer par sexe
direct_sexe <- data.frame(dens = c(valid_g0$pred_direct)
                   , lines = rep(c("Montant prédit"), each = nrow(valid_g0)))
direct_sexe$gender = c(valid_g0$gender)

#Plot.
colnames(valid_g0)
g1 = ggplot(valid_g0) + 
  geom_density(aes(x = pred_direct, fill=gender), alpha = 0.5) +
  labs(y="Densité",x="Montant", fill="Genre") + 
  geom_vline(xintercept=quantile_train_g0[1], linetype="dashed") + 
  geom_vline(xintercept=quantile_train_g0[2], linetype="dashed") + 
  geom_vline(xintercept=quantile_train_g0[3], linetype="dashed") + 
  geom_vline(xintercept=quantile_train_g0[4], linetype="dashed") +
  theme_bw()

g2 = ggplot(valid_g0) + 
  geom_density(aes(x = pred_indirect, fill=gender), alpha = 0.5) +
  labs(y="Densité",x="Montant", fill="Genre") +
  geom_vline(xintercept=quantile_train_g0[1], linetype="dashed") + 
  geom_vline(xintercept=quantile_train_g0[2], linetype="dashed") + 
  geom_vline(xintercept=quantile_train_g0[3], linetype="dashed") + 
  geom_vline(xintercept=quantile_train_g0[4], linetype="dashed") +
  theme_bw()
g2
g3 = ggpubr::ggarrange(g1, g2, ncol=2, nrow=1, common.legend = TRUE, legend="bottom")
#ggsave("./figures/distribution_predictions_quantiles_direct_indirect.pdf",plot=g3, width=10, height=6)


################################################################################
# Fonctions pour les graphiques
################################################################################
NRMSE = function(true, prediction, normalize=TRUE, normalization_method="quantile"){
  #' @param normalization_method (default=quantile). Quantile ou max_min. 
  
  RMSE = sqrt(sum((true-prediction)^2))
 
  if (normalization_method=="quantile"){
    normalization_factor = 1/(quantile(true, 0.75) - quantile(true, 0.25))
  } else if(normalization_method=="max_min")
    normalization_factor = 1/(max(true) - min(true))
  
  if (normalize){
    RMSE = RMSE * normalization_factor
  }
  return(RMSE)
}
# peut-être utiliser une normalisation pour que ce soit plus interprétable

# les quantiles sont calculées sur les données d'entraînement. En espérant que les données d'entraînement représentent bien les données de validation et de test...

quantiles_train_claimcst0 = quantile(train_g0[,"claimcst0"], probs=c(0, 0.25, 0.5, 0.75, 1))

WAGF2 = function(valid, predicted_outcome, true_risk, protected_attribute, protected_attribute_base, ponderation=TRUE, quantiles_train=quantiles_train_claimcst0){
    #' Calculer Weak Actulab Group Fairness définie comme E(M|Q_{\alpha_i} <= R <= Q_{\alpha_{i+1}}, A=a) - E(M|Q_{\alpha_i} <= R <= Q_{\alpha_{i+1}}, A!=a) = \delta. R=true risk=montant réclamé, M=montant réclamé PRÉDIT, A=attribut protégé, e.g. gender.
    #' On calcule la différence de l'espérance pour tous les quantiles et on retourne la moyenne, le max et la médiane.
    #' Suppose que l'attribut protégé a deux modalités (homme/femme) pour simplifier

    x1<-vector(length = (length(quantiles_train)-1))
    for (i in 1:(length(quantiles_train)-1)){
      #TODO
      # en procédant ainsi, les observations qui ont une valeur réelle < le quantile en train ne sont pas considérées dans la métrique. On pourrait choisir de les inclure en conditionnant avec seulement valid$claimcst0 < quantiles_train[[(i+1)]]) lorsque i=1 et seulement valid$claimcst0 >= quantiles_train[[i]] lorsque i=3 (dernier quantile -1)
      if(i==1){
        dat = subset(valid, subset = (valid[, true_risk] < quantiles_train[[(i+1)]]))
      } else if (i==length(quantiles_train)-1){
        dat = subset(valid, subset = (valid[, true_risk] >= quantiles_train[[i]]))
      }else{
        dat = subset(valid, subset = (valid[, true_risk] >= quantiles_train[[i]]) &
                       (valid[, true_risk] < quantiles_train[[(i+1)]]))
      }

      print(paste0("Number of rows in subset ", i, ": ", nrow(dat),"/", nrow(valid)))
      df_1 = subset(dat, subset = dat[,protected_attribute]==protected_attribute_base)
      df_2 = subset(dat, subset = !(dat[,protected_attribute]==protected_attribute_base))
        
      absolute_diff = abs(mean(df_1[,predicted_outcome]) - mean(df_2[,predicted_outcome]))
      if (ponderation){
        x1[i] <- absolute_diff * 1/(quantiles_train[[(i+1)]] - quantiles_train[[i]])
      } else{
        x1[i] <- absolute_diff 
      }
     }
    print(x1)
    delta<-list(maximum=max(x1),mediane=median(x1),moyenne=mean(x1), somme=sum(x1), all_deltas = x1)
}

################################################################################
# Score des modèles direct et indirect, excluant 0
################################################################################

WAGF_direct <- WAGF2(valid_g0,"pred_direct","claimcst0","gender","F",ponderation=T, quantiles_train=quantiles_train_claimcst0)
WAGF_indirect <- WAGF2(valid_g0,"pred_indirect","claimcst0","gender","F",ponderation=T, quantiles_train=quantiles_train_claimcst0)
NRMSE_direct <- NRMSE(valid_g0$claimcst0,valid_g0$pred_direct, normalize=T, normalization_method="max_min")
NRMSE_indirect <- NRMSE(valid_g0$claimcst0,valid_g0$pred_indirect, normalize=T, normalization_method="max_min")

Model <-c("Direct","Indirect","Direct","Indirect")
Value <- c(WAGF_direct$somme, WAGF_indirect$somme, NRMSE_direct, NRMSE_indirect)
#Value <- c( max(WAGF_direct$all_deltas), max(WAGF_indirect$maximum), RMSE_direct, RMSE_indirect)
Metric <- c("somme(PAQ)","somme(PAQ)", "NRMSE","NRMSE")
Metric <- factor(Metric, levels = rev(levels(as.factor(Metric)))) # somme(PAQ) en 1er, NRMSE en 2e, pour le graphique
new_df = data.frame(Model, Value, Metric)



ggplot(new_df, aes(x=Model, y=Value, fill=Metric)) +
    geom_bar(stat = "identity") +
    #coord_cartesian(ylim=c(90000,95000), expand = T) + 
  labs(x="Modèle", y="Valeur", fill="Métrique") + 
  theme_bw()
#ggsave("./figures/NRMSE_sumPAQ_direct_indirect_valid.png", width=8, height=6)












################################################################################
# Analyse des modèles pénalisés
################################################################################
# finalement, on analyse seulement le modèle gamma
# CRÉATION DES DONNÉES INCLUANT LES 0
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

## importer pred gamma 
valid_test_gamma = read.csv(file="./resultats/results_crossval_gamma_linspace_-2_4_20_gamma_WAGF_new_directe.csv") 
# REMARQUE : On va évaluer les modèles seulement sur les observations t.q. claimcst0 > 0, ce qui suppose que le modèle qui identifie s'il y a une réclamation ou non est parfait... Ceci cause un certain biais dans les analyses, mais c'est pour simplifier.

valid_test_gamma$claimcst0 = rbind(valid,test)$claimcst0
valid_test_gamma$gender = rbind(valid,test)$gender

valid_gamma = subset(valid_test_gamma, subset=(which_set==1)) # validation
test_gamma = subset(valid_test_gamma, subset=(which_set==2))

valid_gamma_g0 = subset(valid_test_gamma, subset=(which_set==1 & claimcst0>0)) # validation
test_gamma_g0 = subset(valid_test_gamma, subset=(which_set==2 & claimcst0>0))

# on va combiner test et valid, sinon il n'y aura pas beaucoup d'observations...
#valid_test_gamma_g0 = rbind(valid_gamma_g0, test_gamma_g0)


columns_with_pred = colnames(valid_gamma_g0) %>% str_detect("^X")
lambdas = colnames(valid_gamma_g0) %>% str_extract("\\d+.\\d+")


#### Histogramme des prédictions des montants réclamés
ggplot(valid_gamma_g0) + geom_histogram(aes(x=claimcst0))
ggplot(valid_gamma_g0) + geom_histogram(aes(x=X0.01))
ggplot(valid_gamma_g0) + geom_histogram(aes(x=X526.3252631578947))
ggplot(valid_gamma_g0) + geom_histogram(aes(x=X1052.6405263157894))

summary(valid_gamma_g0$X526.3252631578947)
summary(valid_gamma_g0$X1052.6405263157894)
summary(valid_gamma_g0$X1578.9557894736843)
summary(valid_gamma_g0$X9473.684736842106)

# CONSTAT : certaines valeurs sont beaucoup trop grandes. On va les tronquer à une valeur maximale possible.
maximum_acceptable_prediction = max(train_g0$claimcst0)
valid_gamma_g0_trunc = valid_gamma_g0
for (i in which(columns_with_pred)){
  colonne = valid_gamma_g0_trunc[,i]
  colonne_trunc = ifelse(colonne > maximum_acceptable_prediction, maximum_acceptable_prediction, colonne)
  valid_gamma_g0_trunc[,i] = colonne_trunc
}

# calcul RMSE/equalized odds pour voir l'effet de l'importance
# de la discrimination sur les performances
# DONNÉES DE VALIDATION ET DE TEST CAR SINON JE TROUVE QU'IL N'Y A PAS ASSEZ DE DONNÉES

penalized_models = data.frame()
for (i in which(columns_with_pred)){
    valid_predclaim_i = valid_gamma_g0_trunc[,i]
    
    rmse_i = NRMSE(valid_gamma_g0_trunc$claimcst0, valid_predclaim_i, normalize = TRUE, normalization_method = "max_min")

    WAGF_i = WAGF2(valid_gamma_g0_trunc, colnames(valid_gamma_g0_trunc)[i], "claimcst0", "gender","F", quantiles_train=quantiles_train_claimcst0, ponderation=TRUE)

    values_i = c(rmse_i, WAGF_i$somme, as.numeric(lambdas[i]))
    print(lambdas[i])
    penalized_models = rbind(penalized_models, values_i)
    colnames(penalized_models) = c("NRMSE", "somme(PAQ)", "lambda")
}

penalized_models

################################################################################
# Graphiques time series
################################################################################
valid_g0$pred_direct <- predict( direct, newdata = valid_g0, type = "response")
valid_g0$pred_indirect <- predict( indirect, newdata = valid_g0, type = "response")

WAGF_direct <- WAGF2(valid_g0,"pred_direct","claimcst0","gender","F",ponderation=T, quantiles_train=quantiles_train_claimcst0)
WAGF_indirect <- WAGF2(valid_g0,"pred_indirect","claimcst0","gender","F",ponderation=T, quantiles_train=quantiles_train_claimcst0)
NRMSE_direct <- NRMSE(valid_g0$claimcst0,valid_g0$pred_direct, normalize=T, normalization_method="max_min")
NRMSE_indirect <- NRMSE(valid_g0$claimcst0,valid_g0$pred_indirect, normalize=T, normalization_method="max_min")

colors <- c("NRMSE" = hue_pal()(2)[1], "somme(PAQ)" = hue_pal()(2)[2])
p1 = ggplot(penalized_models) + 
    geom_line(aes(x=lambda, y=NRMSE, color="NRMSE")) + 
    geom_point(aes(x=lambda, y=NRMSE, color="NRMSE")) +
    geom_line(aes(x=lambda, y=`somme(PAQ)`, color="somme(PAQ)")) +
    geom_point(aes(x=lambda, y=`somme(PAQ)`, color="somme(PAQ)")) +
    scale_color_manual(values = colors) +
    #scale_y_continuous("RMSE", sec.axis = sec_axis(~ . /1000, name="WAGF"))+
    geom_point(aes(x=0, y=WAGF_direct$somme), colour=colors[2],shape=3, size=3) +
    geom_point(aes(x=0, y=NRMSE_direct), colour=colors[1],shape=3,size=3) +
    geom_point(aes(x=0, y=WAGF_indirect$somme), colour=colors[2], shape=4,size=3) +
    geom_point(aes(x=0, y=NRMSE_indirect), colour=colors[1], shape=4, size=3) +
    geom_vline(xintercept=penalized_models$lambda[10], linetype="dashed", color="black") +
    labs(title="Effet de la pénalité sur l'équité et la performance (validation)", x=TeX("$\\lambda$"), y="Métrique", color="Métrique") +
    theme_bw()
# "+" = directe, "x" = indirecte. Le mettre en notes de bas de page dans le beamer car je n'ai pas trouvé comment l'ajouter dans le graphique.
#ggsave("./figures/performance_equite_gamma.pdf", plot=p1, width=10,height=6)


################################################################################
# Diagramme à bandes sur les données de test
################################################################################
test_g0$pred_direct <-  predict( direct, newdata = test_g0, type = "response")
test_g0$pred_indirect <-predict( indirect, newdata = test_g0, type = "response")
test_g0$pred_penalized <- test_gamma_g0[,11]

WAGF_direct <- WAGF2(test_g0,"pred_direct","claimcst0","gender","F",ponderation=T, quantiles_train=quantiles_train_claimcst0)
WAGF_indirect <- WAGF2(test_g0,"pred_indirect","claimcst0","gender","F",ponderation=T, quantiles_train=quantiles_train_claimcst0)
WAGF_penalized <- WAGF2(test_g0,"pred_penalized","claimcst0","gender","F",ponderation=T, quantiles_train=quantiles_train_claimcst0)

NRMSE_direct <- NRMSE(test_g0$claimcst0,test_g0$pred_direct, normalize=T, normalization_method="max_min")
NRMSE_indirect <- NRMSE(test_g0$claimcst0,test_g0$pred_indirect, normalize=T, normalization_method="max_min")
NRMSE_penalized <- NRMSE(test_g0$claimcst0,test_g0$pred_penalized, normalize=T, normalization_method="max_min")





Model <-c("Direct","Indirect", "Penalisé", "Direct","Indirect", "Penalisé")
Value <- c(WAGF_direct$somme, WAGF_indirect$somme, WAGF_penalized$somme, NRMSE_direct, NRMSE_indirect, NRMSE_penalized)
#Value <- c( max(WAGF_direct$all_deltas), max(WAGF_indirect$maximum), RMSE_direct, RMSE_indirect)
Metric <- c("somme(PAQ)","somme(PAQ)","somme(PAQ)", "NRMSE","NRMSE","NRMSE")
Metric <- factor(Metric, levels = rev(levels(as.factor(Metric)))) # somme(PAQ) en 1er, NRMSE en 2e, pour le graphique
new_df = data.frame(Model, Value, Metric)



ggplot(new_df, aes(x=Model, y=Value, fill=Metric)) +
  geom_bar(stat = "identity") +
  #coord_cartesian(ylim=c(90000,95000), expand = T) + 
  labs(x="Modèle", y="Valeur", fill="Métrique") + 
  theme_bw()
#ggsave("./figures/NRMSE_sumPAQ_direct_indirect_test.png", plot=last_plot(), width=8, height=6)


# finalement, distribution des prédictions à travers les quantiles
g1 = ggplot(test_g0) + 
  geom_density(aes(x = pred_direct, fill=gender), alpha = 0.5) +
  labs(y="Densité",x="Montant", fill="Genre") + 
  geom_vline(xintercept=quantile_train_g0[1], linetype="dashed") + 
  geom_vline(xintercept=quantile_train_g0[2], linetype="dashed") + 
  geom_vline(xintercept=quantile_train_g0[3], linetype="dashed") + 
  geom_vline(xintercept=quantile_train_g0[4], linetype="dashed") +
  theme_bw()

g2 = ggplot(test_g0) + 
  geom_density(aes(x = pred_indirect, fill=gender), alpha = 0.5) +
  labs(y="Densité",x="Montant", fill="Genre") +
  geom_vline(xintercept=quantile_train_g0[1], linetype="dashed") + 
  geom_vline(xintercept=quantile_train_g0[2], linetype="dashed") + 
  geom_vline(xintercept=quantile_train_g0[3], linetype="dashed") + 
  geom_vline(xintercept=quantile_train_g0[4], linetype="dashed") +
  theme_bw()

g3 = ggplot(test_g0) + 
  geom_density(aes(x = pred_penalized, fill=gender), alpha = 0.5) +
  labs(y="Densité",x="Montant", fill="Genre") +
  geom_vline(xintercept=quantile_train_g0[1], linetype="dashed") + 
  geom_vline(xintercept=quantile_train_g0[2], linetype="dashed") + 
  geom_vline(xintercept=quantile_train_g0[3], linetype="dashed") + 
  geom_vline(xintercept=quantile_train_g0[4], linetype="dashed") +
  theme_bw()

g4 = ggpubr::ggarrange(g1, g2, g3, ncol=3, nrow=1, common.legend = TRUE, legend="bottom")
#ggsave("./figures/distribution_predictions_quantiles_direct_indirect_penalized_test.pdf",plot=g4, width=14, height=6)

