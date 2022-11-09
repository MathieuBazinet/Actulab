library(tidyverse);library(MASS)

df = read.csv("dataCar.csv")
df = df %>%
  select(-X_OBSTAT_) %>%
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

round(df$numclaims %>% table() %>% prop.table()*100,2)

################################################################################
# Attributs protégés
################################################################################
# gender
# agecat

################################################################################
# exposition
################################################################################
# la variable exposure correspond à la proportion de l'année pendant
# laquelle l'assuré(e) est couvert(e).
# (http://cas.uqam.ca/pub/web/CASdatasets-manual.pdf)
# Plus l'individu est assuré # longtemps, plus il y a possiblement de 
# réclamations (proportionnel, # donc on va avoir une variable d'offset)
# Nous incluons cette variable comme un terme d'offset.
# Nous créons donc la variable log_Exposure
df$log_exposure <- log(df$exposure) # log=ln

################################################################################
# Poisson ou binomiale négative?
################################################################################
formule = formula(numclaims~veh_value+veh_body+veh_age+area+offset(log_exposure))

test_surdispersion = function(formula, dataset){
  poisson = glm(formula, family=poisson(link="log"), data=dataset)
  bn = glm.nb(formula, data=dataset)
  
  1/2*(1-pchisq(bn$deviance-poisson$deviance, df=1, lower.tail=F))
}

### Poisson
test_surdispersion(formula, df)
# on rejette H0 donc il y a de l'évidence qu'il y a de la surdispersion.


################################################################################
# Modèle de Poisson
################################################################################
poisson = glm(formula, family=poisson(link="log"), data=df)
summary(poisson)

################################################################################
# Équité dans le modèle de Poisson
################################################################################
library(fairness)