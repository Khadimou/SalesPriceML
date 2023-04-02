# SalesPriceML
Quel est le prix de vente idéal pour des maisons d'une agence immobilière ? 

Le but de ce projet est de prédire le prix de vente d'un bien immobilier en fonctions des différentes caractéristiques de la maison. 


Le projet comprend les 4 phases suivantes:

## Analyse exploratoire de données
Dans cette phase, il est question d'analyser les données, d'identifier les relations entre les différentes variables. 
* Features selection
Ici, nous sélectonnons les différentes caractéristiques qui contribuent significativement dans la prédiction de la variable cible **SalePrice**
## Feature selection
Les méthodes suivantes ont été utilisées dans cette phase:
* Corrélation entre les caractéristiques et la target
* RandomForestRegressor pour calculer l'importance des features
* Lasso Regression
* Recursive Feature Elimination
* Mutual information Feature
* Anova pour selectionner les features catégorielles 
## modélisation 
Dans cette phase, nous avons entrainé 5 algorithmes de regression:
* RandomForestRegressor
* GradientBoostingRegressor
* HistGradientBoostingRegressor
* ExtraTreesRegressor
* SVR
## Déploiement 
Dans cette section, nous avons exposé nos modèles dans une application web construite avec **Streamlit**. 
La web application est composée des sections suivanes:
 ### Home 
Ici, l'utilisateur peut prédire le prix de vente en choisissant ses critères sur 4 caractéristiques
Il peut également avoir un aperçu sur les données utilisées dans ce projet
### EDA
Ici, l'utilisateur a le choix de visualiser la target SalePrice en fonction des features avec :
* Un boxplot
* un Scatter 
* Une lineplot
Il peut également visualiser la distribution de chaque paramètre. 
Il a également la possibilité de nettoyer les donnnées avec la méthode de l'écart interquartile. 
### Feature Engineering
Cette section permet à l'utilisateur de selectionner les features selon les différentes présentées ci-haut. 
Pour cela, il voit en temps réel la liste des features sélectionnées. 
### Modélisation
Cette section permet à l'utilisateur d'entrainer le modèle de son choix et les features de son choix. 
Aprése entrainement, il voit:
* Les métriques d'entrainement et de test
* le score de cross_validation aprés 10 runs 







