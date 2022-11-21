library(tidyverse);library(MASS);library(fairness);library(ggplot2)

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
#write.csv(df, "dataCar_clean.csv",row.names=FALSE)

df_claimsg0 <- subset(df, subset=claimcst0>0) # df sans les motants nuls 

ids_split <- splitTools::partition(
    y = df_claimsg0[, "clm"]
    ,p = c(train = 0.7, valid = 0.15, test = 0.15)
    ,type = "stratified" # stratified is the default
    ,seed = 42
)
train <- df_claimsg0[ids_split$train, ]
valid <- df_claimsg0[ids_split$valid, ]
test <- df_claimsg0[ids_split$test, ]

train$which_set = 0
valid$which_set = 1
test$which_set = 2


################################################################################
# Régression gamma
################################################################################
# juste pour voir le genre de sortie et les paramètres estimés par la fonction
# df_claimsg0 = subset(train, subset=claimcst0>0)

gam = glm(claimcst0 ~ gender + area, data=train, family = Gamma(link="log"))
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
# Régression gamma directe et indirecte 
################################################################################
# Direct 
direct = glm(claimcst0 ~ veh_value+veh_body+veh_age+gender+area, data=train, family = Gamma(link="log"))
summary(direct)

# Indirect

indirect = glm(claimcst0 ~ veh_value+veh_body+veh_age  + area, data=train, family = Gamma(link="log"))
summary(indirect)


valid$pred_direct <- predict( direct, newdata = valid, type = "response")
valid$pred_indirect <- predict( indirect, newdata = valid, type = "response")

test$pred_direct <-  predict( direct, newdata = test, type = "response")
test$pred_indirect <-predict( indirect, newdata = test, type = "response")


################################################################################
# Analyse de la distribution des montants réclamés à l'intérieur des quantiles
# on ne veut pas avoir slm des hommes dans le 3e-4e quantile, p.ex.
################################################################################

## pas très élaboré comme code mais ça fait la job 
## ech d'entrainement 
quantile_train <- quantile(train$claimcst0)
train_q1 <- subset(train, subset = (train$claimcst0 < quantile_train[[2]]) ) 
summary(train_q1)

train_q2 <- subset(train, subset = ((train$claimcst0 >= quantile_train[[2]]) &
                                        (train$claimcst0 < quantile_train[[3]]))) 
summary(train_q2)

train_q3 <- subset(train, subset = ((train$claimcst0 >= quantile_train[[3]]) &
                                        (train$claimcst0 < quantile_train[[4]]))) 
summary(train_q3)

train_q4 <- subset(train, subset = ((train$claimcst0 >= quantile_train[[4]]))) 
summary(train_q4)

par(mfrow=c(2,2))
boxplot_q1 <-boxplot(train_q1$claimcst0~train_q1$gender, col=c("#009999", "#0000FF"), xlab = "Genre", ylab = "Montant réclamé", main="[min,Q1]")
boxplot_q2 <-boxplot(train_q2$claimcst0~train_q2$gender, col=c("#009999", "#0000FF"), xlab = "Genre", ylab = "Montant réclamé", main="[Q1,Q2]")
boxplot_q3 <-boxplot(train_q3$claimcst0~train_q3$gender, col=c("#009999", "#0000FF"), xlab = "Genre", ylab = "Montant réclamé", main="[Q3,Q4]")
boxplot_q4 <-boxplot(train_q4$claimcst0~train_q4$gender, col=c("#009999", "#0000FF"), xlab = "Genre", ylab = "Montant réclamé", main="[Q4,max]")
mtext("Distribution du montant réclamé dans les différents quantiles selon le genre", side = 3, line = - 1,font=2, outer = TRUE)
par(mfrow=c(1,1))

## ech de validation 
quantile_valid <- quantile(valid$claimcst0)
valid_q1 <- subset(valid, subset = (valid$claimcst0 < quantile_valid[[2]]) ) 
summary(valid_q1)

valid_q2 <- subset(valid, subset = ((valid$claimcst0 >= quantile_valid[[2]]) &
                                        (valid$claimcst0 < quantile_valid[[3]]))) 
summary(valid_q2)

valid_q3 <- subset(valid, subset = ((valid$claimcst0 >= quantile_valid[[3]]) &
                                        (valid$claimcst0 < quantile_valid[[4]]))) 
summary(valid_q3)

valid_q4 <- subset(valid, subset = ((valid$claimcst0 >= quantile_valid[[4]]))) 
summary(valid_q4)

par(mfrow=c(2,2))
boxplot_q1 <-boxplot(valid_q1$claimcst0~valid_q1$gender, col=c("#009999", "#0000FF"), xlab = "Genre", ylab = "Montant réclamé", main="[min,Q1]")
boxplot_q2 <-boxplot(valid_q2$claimcst0~valid_q2$gender, col=c("#009999", "#0000FF"), xlab = "Genre", ylab = "Montant réclamé", main="[Q1,Q2]")
boxplot_q3 <-boxplot(valid_q3$claimcst0~valid_q3$gender, col=c("#009999", "#0000FF"), xlab = "Genre", ylab = "Montant réclamé", main="[Q3,Q4]")
boxplot_q4 <-boxplot(valid_q4$claimcst0~valid_q4$gender, col=c("#009999", "#0000FF"), xlab = "Genre", ylab = "Montant réclamé", main="[Q4,max]")
mtext("Distribution du montant réclamé dans les différents quantiles selon le genre", side = 3, line = - 1,font=2, outer = TRUE)
par(mfrow=c(1,1))



################################################################################
# Analyse de la distribution des montants réclamés prédits par les différents
# modèles
################################################################################


#Modèle direct
dat1 <- data.frame(dens = c(valid$claimcst0, valid$pred_direct)
                   , lines = rep(c("Montant reclamé", "Montant prédit"), each = nrow(valid)))
#Plot.
ggplot(dat1, aes(x = dens, fill = lines)) + 
    geom_density(alpha = 0.5) +
    labs(y="Densité",x="Montant", title="Distribution des prédictions : Modèle direct")

#Modèle indirect
dat2 <- data.frame(dens = c(valid$claimcst0, valid$pred_indirect)
                   , lines = rep(c("Montant reclamé", "Montant prédit"), each = nrow(valid)))
#Plot.
ggplot(dat2, aes(x = dens, fill = lines)) + 
    geom_density(alpha = 0.5) +
    labs(y="Densité",x="Montant", title="Distribution des prédictions : Modèle indirect")


################################################################################
# Fonctions pour les graphiques
################################################################################
RMSE = function(true, prediction){
    sqrt(sum((true-prediction)^2))
}
# peut-être utiliser une normalisation pour que ce soit plus interprétable

# les quantiles sont calculées sur les données d'entraînement. En espérant que les données d'entraînement représentent bien les données de validation et de test...
quantiles_train = quantile(train$numclaims, probs=)

WAGF = function(train, valid, predicted_outcome, true_risk, protected_attribute, protected_attribute_base, quantiles=c(0,0.25,0.5,0.75,1)){
    #' Calculer Weak Actulab Group Fairness définie comme E(M|Q_{\alpha_i} <= R <= Q_{\alpha_{i+1}}, A=a) - E(M|Q_{\alpha_i} <= R <= Q_{\alpha_{i+1}}, A!=a) = \delta. R=true risk=montant réclamé, M=montant réclamé PRÉDIT, A=attribut protégé, e.g. gender.
    #' On calcule la différence de l'espérance pour tous les quantiles et on retourne la moyenne, le max et la médiane.
    #' Suppose que l'attribut protégé a deux modalités (homme/femme) pour simplifier
    
    quantiles_train = quantile(train[,true_risk], probs=quantiles)
    
    df_1 = valid[valid[,protected_attribute]==protected_attribute_base & valid[,predicted_outcome] %in% quantiles_train]
    df_2 = valid[valid[,protected_attribute]!=protected_attribute_base & valid[,predicted_outcome] %in% quantiles_train]
    
    mean(df1[,predicted_outcome]) - mean(df2[,predicted_outcome])
    
}


## je ne veux pas modifier ton code alors je fais une nouvelle fonction WAGF2
## quantile de numclaim ou claimst0 ??

WAGF2 = function(train, valid, predicted_outcome, true_risk, protected_attribute, protected_attribute_base, quantiles=c(0,0.25,0.5,0.75,1)){
    #' Calculer Weak Actulab Group Fairness définie comme E(M|Q_{\alpha_i} <= R <= Q_{\alpha_{i+1}}, A=a) - E(M|Q_{\alpha_i} <= R <= Q_{\alpha_{i+1}}, A!=a) = \delta. R=true risk=montant réclamé, M=montant réclamé PRÉDIT, A=attribut protégé, e.g. gender.
    #' On calcule la différence de l'espérance pour tous les quantiles et on retourne la moyenne, le max et la médiane.
    #' Suppose que l'attribut protégé a deux modalités (homme/femme) pour simplifier
    
    quantiles_train = quantile(train[,true_risk], probs=quantiles)
    x1<-vector(length = (length(quantiles_train)-1))
    
    for (i in 1:(length(quantiles_train)-1)){
        
        dat = subset(valid, subset = (valid$claimcst0 >= quantiles_train[[i]]) &
                           (valid$claimcst0 < quantiles_train[[(i+1)]]))
        
        df_1 = subset(dat, subset =dat[,protected_attribute]==protected_attribute_base)
        df_2 = subset(dat, subset =! (dat[,protected_attribute]==protected_attribute_base))
        
        x1[i] <- abs(mean(df_1[,predicted_outcome]) - mean(df_2[,predicted_outcome]))
        
        
    }
    
    delta<-list(maximum=max(x1),mediane=median(x1),moyenne=mean(x1), all_deltas = x1)
    
}


WAGF_direct <- WAGF2(train,valid,"pred_direct","claimcst0","gender","F",quantiles=c(0,0.25,0.5,0.75,1))
WAGF_indirect <- WAGF2(train,valid,"pred_indirect","claimcst0","gender","F",quantiles=c(0,0.25,0.5,0.75,1))
RMSE_direct <- RMSE(valid$claimcst0,valid$pred_direct)
RMSE_indirect <- RMSE(valid$claimcst0,valid$pred_indirect)

Model <-c("Direct","Indirect","Direct","Indirect")
Value <- c( WAGF_direct$maximum, WAGF_indirect$maximum, RMSE_direct, RMSE_indirect)
Metric <-c("delta","delta","RMSE","RMSE")
new_df = data.frame(Model, Value,Metric)


ggplot(new_df, aes(x=Model, y=Value, fill=Metric)) +
    geom_bar(stat = "identity") +
    coord_cartesian(ylim=c(90000,94000), expand = F)
















################################################################################
# Analyse des modèles pénalisés
################################################################################


### retour au jeu de donnée avec 0
ids_split0 <- splitTools::partition(
    y = df[, "clm"]
    ,p = c(train0 = 0.7, valid0 = 0.15, test0 = 0.15)
    ,type = "stratified" # stratified is the default
    ,seed = 42
)
train0 <- df[ids_split0$train0, ]
valid0 <- df[ids_split0$valid0, ]
test0 <- df[ids_split0$test0, ]

train0$which_set = 0
valid0$which_set = 1
test0$which_set = 2

## importer prob logistique 
valid_test_probs = read.csv(file="results_crossval_gamma_linspace_-2_4_20_binomial_EO.csv") 

valid_probs = subset(valid_test_probs, which_set==1) # validation
valid_probs$clm = valid0$clm
valid_probs$gender = valid0$gender

columns_with_probs = colnames(valid_probs) %>% str_detect("^X")
lambdas = colnames(valid_probs) %>% str_extract("\\d+.\\d+")

cutoff = mean(train0$clm)

## importer pred gamma 
valid_test_gamma = read.csv(file="results_crossval_gamma_linspace_-2_4_20_gamma_WAGF.csv") 

valid_predclaim = subset(valid_test_gamma, which_set==1) # validation
valid_predclaim$claimcst0 = valid0$claimcst0
valid_predclaim$gender = valid0$gender


columns_with_pred = colnames(valid_predclaim) %>% str_detect("^X")
lambdas = colnames(valid_predclaim) %>% str_extract("\\d+.\\d+")

# calcul RMSE/equalized odds pour voir l'effet de l'importance
# de la discrimination sur les performances
# DONNÉES DE VALIDATION

penalized_models = data.frame()


for (i in which(columns_with_probs)){
    pred_clm_i = ifelse(valid_probs[,i]>cutoff,1,0)
    pred_claimcst0_i = valid_predclaim[,i]
    pred_logXgam_i = pred_clm_i*pred_claimcst0_i
    
    rmse_i = RMSE(valid_predclaim$claimcst0,pred_logXgam_i)
    
    data<-subset(train0,subset = (train0$claimcst0>0))  ### retirer 0 pour quantiles ?
    
    WAGF_i = WAGF2(data,valid_predclaim,i ,"claimcst0","gender","F",quantiles=c(0,0.25,0.5,0.75,1))
    
    values_i = c(rmse_i, WAGF_i$maximum,as.numeric(lambdas[i]))
    print(lambdas[i])
    penalized_models = rbind(penalized_models, values_i)
    colnames(penalized_models) = c("RMSE", "WAGF", "lambda")
    
}


colors <- c("RMSE" = hue_pal()(2)[1], "WAGF" = hue_pal()(2)[2])
p1 = ggplot(penalized_models) + 
    geom_line(aes(x=lambda, y=RMSE, color="RMSE")) + 
    geom_point(aes(x=lambda, y=RMSE, color="RMSE")) +
    geom_line(aes(x=lambda, y=WAGF, color="WAGF")) +
    geom_point(aes(x=lambda, y=WAGF, color="WAGF")) +
    scale_color_manual(values = colors) +

    scale_y_continuous("RMSE", sec.axis = sec_axis(~ . /10000, name="WAGF"))+

    labs(title="Effet de la pénalité sur l'équité et la performance", x=TeX("$\\lambda$"), color="Métrique") +
    theme_bw()

p1


### TODO visualisation




