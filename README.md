# Travail fait dans le cadre d'Actulab

L'entièreté du code ici présent a été codé dans le cadre de l'atelier d'innovation Actulab.
Avec mes collègues Simon-Olivier Lépine, Liliane Ouedraogo, Michaël Rioux et moi-même, 
Mathieu Bazinet, nous avons décidé de proposer une réflexion sur l'équité en assurance. 

Vous trouverez notre implémentation d'un modèle linéaire généralisé pénalisé par une mesure
d'équité. Les distributions présentement supportées sont la régression logistique, la régression
poisson ainsi que la régression gamma. Nous proposons d'entraîner le modèle sans pénalité, 
en pénalisant avec la parité démographique, la disparité moindres ainsi que la parité 
$\textcolor{red}{\text{Actulab}}$ proposée durant notre présentation. 

Le modèle _main.py_ permet de lancer l'entraînement d'un modèle avec sur la base de données
dataCar tout en protégeant un ou plusieurs attributs. Le modèle se trouve dans le fichier
_Fairness_aware_model.py_. L'exploration des résultats a été faite en R.

Un fichier requirements.yml est fourni avec le code. Statsmodel pose problème avec certaines
versions de pandas. Vous pouvez créer un nouvel environnement qui fonctionne avec : 
```
conda env create -n ENVNAME --file requirements.yml
```



