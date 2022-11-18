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