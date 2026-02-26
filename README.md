# Projet Final - PSY3019
*Océane Vallée, 20204008*

## Description du projet
Le projet que j'ai choisit est celui de Wei-Xuan Chai (2025) intitulé *Decoding Depression via EEG Biomarkers: A Neurocomputational Approach using Machine and Deep Learning*. 
L'objectif de l'auteur pour ce projet était d'explorer la capacité de deux différents modèles d'intelligence artificielle à classifier correctement le trouble depressif majeur via des caractéristiques EGG du domaine temporel. 

### Modèles d'IA utilisés
Le premier modèle à avoir été utilisé est une machine à vecteurs de support (SVM) qui est un algorithme d'apprentissage automatique supervisé principalement utilisé pour la classification et la régression. 

Le deuxième modèle utilisé par l'auteur est un modèle d'apprentissage profond (DL) basé sur EEGNet. Comme opérer un modèle DL demande beaucoup de ressources, un prototype fonctionnel à été implémenté sur GoogleColab. 

### Données
La base de données utilisé pour ce projet contient 30 participants contrôles (H) et 34 participants ayant un trouble dépresif majeur (MDD). Pour les 64 participants, trois types de données sont disponibles : 
1. Données d'enregistrement EEG avec les yeux fermés (EC)
2. Données d'enregistrement EEG avec les yeux ouverts (EO)
3. Données d'enregistrement EEG de la condition P300 (TASK)

Dans le cadre du projet, l'auteur a utilisé seulement les 5 premières minutes des données EEG avec les yeux fermés. De plus, il n'a pas réussit à télécharger les données de tous les participants de la base de données. Ainsi, les analyses ont été effectuées sur 28 H et 30 MDD. 

### Résultats
- Le modèle de SVM à atteint une performance d'environ 95%
- Le modèle de DL à atteint une performance d'environ 80%

*L'auteur mentionne que même si l'on devrait s'attendre à ce que le DL soit plus performant que le SVM, ces résultats sont probablement lié au réglage limité des hyperparamètres et une optimisation incomplète du modèle de DL.*

<img width="576" height="432" alt="image" src="https://raw.githubusercontent.com/OceaneVallee/brainhack-EEG-depression-ml-dl/refs/heads/main/cover%20img.png" />

## Pourquoi ce projet

J'ai choisit ce projet, parce que je voulais me familiariser avec le traitement de données EEG pour mon projet de Honor sur l'imagerie mentale, ainsi que pour mon projet de maîtrise  l'an prochain sur la conscience. 

De plus, je suis présentement le cours d'IA avec monsieur Shahab Bakhtiari, donc je trouvais que ce projet était une bonne façon de transposé mes nouvelles connaissances. 

## Tâches 

### Tâche 1 : Restructuration des données
- Restructurer et simplifier le code
- Ajouter des explications afin d'améliorer la compréhension du code
- Traduire les lignes de commentaire écrites en mandarin
- Ajouter un fichier "environnement.yml" afin de facilité la reproductibilité du projet


### Tâche 2 : Reproductibilité 
- Reproduction des notebooks existant
- Documentation du processus de reproduction
- Correction des bugs s'il-y-a lieu


### Tâche 3 : Extension et validation des analyses
- Tester le code sur les données EO et TASK
- Pré-traitement des données EO et TASK

*Si cette tâche s’avère non réalisable, elle sera remplacée par l’automatisation du projet et par une justification méthodologique de la non-applicabilité du code aux autres données.*

## Références
Chai, W.-X. (2025, 15 juin). brainhack-EEG-depression-ml-dl [Code source]. GitHub. https://github.com/ChaiWeiXuan/brainhack-EEG-depression-ml-dl

Mumtaz, W. (2016, 23 novembre). MDD Patients and Healthy Controls EEG Data (New). figshare. doi:10.6084/m9.figshare.4244171.v2 
