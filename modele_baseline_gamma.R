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
boxplot_q1 <-boxplot(train_q1$claimcst0~train_q1$gender, col=c("pink", "Blue"))
boxplot_q2 <-boxplot(train_q2$claimcst0~train_q2$gender, col=c("pink", "Blue"))
boxplot_q3 <-boxplot(train_q3$claimcst0~train_q3$gender, col=c("pink", "Blue"))
boxplot_q4 <-boxplot(train_q4$claimcst0~train_q4$gender, col=c("pink", "Blue"))
par(mfrow=c(1,1))


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
    
    delta<-c(Maximum=max(x1),Mediane=median(x1),Moyenne=mean(x1))
    print(delta)
    
}


WAGF_direct <- WAGF2(train,valid,"pred_direct","claimcst0","gender","F",quantiles=c(0,0.25,0.5,0.75,1))
WAGF_indirect <- WAGF2(train,valid,"pred_indirect","claimcst0","gender","F",quantiles=c(0,0.25,0.5,0.75,1))
RMSE_direct <- RMSE(valid$claimcst0,valid$pred_direct)
RMSE_indirect <- RMSE(valid$claimcst0,valid$pred_indirect)

Model <-c("Direct","Indirect","Direct","Indirect")
Value <- c( WAGF_direct[[1]], WAGF_indirect[[1]], RMSE_direct, RMSE_indirect)
Metric <-c("delta","delta","RMSE","RMSE")
new_df = data.frame(Model, Value,Metric)


ggplot(new_df, aes(x=Model, y=Value, fill=Metric)) +
    geom_bar(stat = "identity") +
    coord_cartesian(ylim=c(90000,94000), expand = F)

