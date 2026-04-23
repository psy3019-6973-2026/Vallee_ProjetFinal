# Projet Final - PSY3019
*Océane Vallée, 20204008*

## Description du projet
Le projet que j'ai choisi est celui de Wei-Xuan Chai (2025) intitulé *Decoding Depression via EEG Biomarkers: A Neurocomputational Approach using Machine and Deep Learning*. 
L'objectif de l'auteur pour ce projet était d'explorer la capacité de deux différents modèles d'intelligence artificielle à classifier correctement le trouble dépressif majeur via des caractéristiques EEG du domaine temporel. 

### Modèles d'IA utilisés
Le premier modèle à avoir été utilisé est une machine à vecteurs de support (SVM) qui est un algorithme d'apprentissage automatique supervisé principalement utilisé pour la classification et la régression. 

Le deuxième modèle utilisé par l'auteur est un modèle d'apprentissage profond (DL) basé sur EEGNet. Comme opérer un modèle DL demande beaucoup de ressources, un prototype fonctionnel a été implémenté sur Google Colab. 

### Données
La base de données utilisée pour ce projet contient 30 participants contrôles (H) et 34 participants ayant un trouble dépressif majeur (MDD). 

Pour les 64 participants, trois types de données sont disponibles : 
1. Données d'enregistrement EEG avec les yeux fermés (EC)
2. Données d'enregistrement EEG avec les yeux ouverts (EO)
3. Données d'enregistrement EEG de la condition P300 (TASK)

Dans le cadre du projet, l'auteur a utilisé seulement les 5 premières minutes des données EEG avec les yeux fermés. De plus, il n'a pas réussi à télécharger les données de tous les participants de la base de données. Ainsi, les analyses ont été effectuées sur 28 H et 30 MDD. 

### Résultats
- Le modèle de SVM a atteint une performance d'environ 95 %
- Le modèle de DL a atteint une performance d'environ 80 %

*L'auteur mentionne que même si l'on devrait s'attendre à ce que le DL soit plus performant que le SVM, ces résultats sont probablement liés au réglage limité des hyperparamètres et à une optimisation incomplète du modèle de DL.*

<img width="576" height="432" alt="image" src="https://raw.githubusercontent.com/OceaneVallee/brainhack-EEG-depression-ml-dl/refs/heads/main/cover%20img.png" />

## Pourquoi ce projet

J'ai choisi ce projet, parce que je voulais me familiariser avec le traitement de données EEG pour mon projet de Honor sur l'imagerie mentale, ainsi que pour mon projet de maîtrise l'an prochain sur la conscience. 

De plus, je suis présentement le cours d'IA avec monsieur Shahab Bakhtiari, donc je trouvais que ce projet était une bonne façon de transposer mes nouvelles connaissances. 

## Tâches 

### Tâche 1 : Reproductibilité 
- Reproduction des notebooks existants
- Documentation du processus de reproduction
- Correction des bugs s'il y a lieu


### Tâche 2 : Restructuration des données
- Restructurer et simplifier le code
- Ajouter des explications afin d'améliorer la compréhension du code
- Traduire les lignes de commentaire écrites en mandarin
- Ajouter un fichier "environnement.yml" afin de faciliter la reproductibilité du projet


### Tâche 3 : Extension et validation des analyses
- Tester le code sur les données EO et TASK
- Pré-traitement des données EO et TASK
- Essayer de diagnostiquer les performances suspicieusement élevées de l'auteur grâce aux nouvelles données

*Si cette dernière tâche s’avère non réalisable, elle sera remplacée par l’automatisation du projet et par une justification méthodologique de la non-applicabilité du code aux autres données.*

## Travail éffectué

### Tâche 1 (10,17 heures de travail)
- Reproduire le notebook 1 pour assurer une bonne compréhension du code (1,67h)
- Reproduire le notebook 2 pour assurer une bonne compréhension du code (0,17h)
- Reproduire le notebook 3 pour assurer une bonne compréhension du code (1,33h)
- Traduire les lignes de commentaires en mandarin (0,17h)
- Créer un fichier environnement.yml (1,25h)
- Automatisation du code : restructuration du dépôt GitHub, ajout des fichiers nécessaires au bon fonctionnement d'invoke et débogage du code (5,58h)

### Tâche 2 (6,34 heures de travail)
- Restructuration et simplicifation du code, ajout de lignes explicatives et résumé de la pipeline du notebook 1 (2,59h)
- Restructuration et simplicifation du code, ajout de lignes explicatives et résumé de la pipeline du notebook 2 (1h)
- Restructuration et simplicifation du code, ajout de lignes explicatives et résumé de la pipeline du notebook 3 (2,75h)

### Tâche 3 (# heures de travail)
- Prétraitement des deux autres conditions (EO et Task) de la base de données (1,75h)
- Ajustement notebooks de ML (2 et 3) pour pouvoir traiter les données complètes (avec les nouvelles conditons) (0,25h)
- Ajustement des lignes explicatives en fonctions des modifications apportées aux notebooks (0,33h)
- Test du code sur les nouvelles données et débogage (0,8h)
- Vérification en profondeur des modèles de ML (notebook 2) pour comprendre les performances anormalement hautes et résoudre les problèmes de data leakage (0,53h)
- Documentation du processus et explications (#h)

## Reproduction du projet

### Prérequis

- [Git](https://git-scm.com/)
- [Conda](https://docs.conda.io/en/latest/miniconda.html) (recommandé) **ou** Python ≥ 3.10 avec pip
- [Invoke](https://www.pyinvoke.org/) (`pip install invoke`)

---

### 1. Cloner le dépôt

```bash
git clone https://github.com/psy3019-6973-2026/Vallee_ProjetFinal.git
cd Vallee_ProjetFinal
```

---

### 2. Créer l'environnement

#### Option A — Conda (recommandé)

```bash
conda env create -f environnement.yml
conda activate projet_final
```

#### Option B — pip

```bash
pip install -r requirements.txt
```

---

### 3. Télécharger les données

Les données EEG proviennent de Figshare ([Mumtaz, 2016](https://doi.org/10.6084/m9.figshare.4244171.v2)) et sont téléchargées automatiquement via :

```bash
invoke fetch
```

Les fichiers seront enregistrés dans le dossier `source_data/4244171/`.

---

### 4. Reproduire les analyses

Les notebooks peuvent être exécutés individuellement dans Jupyter, ou automatiquement via les commandes `invoke` suivantes :

| Commande | Description |
|---|---|
| `invoke run-preprocessing` | Prétraitement EEG et extraction des bandpowers (`1_preprocessing_bandpower.ipynb`) |
| `invoke run-svm` | Classification par SVM (`2_ml_svm.ipynb`) — exécute aussi le prétraitement |
| `invoke run` | Pipeline complet : prétraitement → SVM |

> **Note :** Le notebook `3_dl_eegnet.ipynb` (modèle EEGNet) doit être exécuté manuellement dans [Google Colab](https://colab.research.google.com/). Il n'est pas inclus dans le pipeline automatisé, car l'entraînement d'un modèle de deep learning demande des ressources trop importantes.

---

### 5. Résultats

Les fichiers générés (figures `.png`, données `.csv`, poids du modèle `.h5`) sont sauvegardés dans le dossier `output_data/`.

Pour supprimer les résultats générés :

```bash
invoke clean
```

## Comparaison des résultats : original vs. corrigé

### Différences méthodologiques

| | Projet original (Chai, 2025) | Projet corrigé (Vallée, 2026) |
|---|---|---|
| **Participants** | 28 contrôles (H) + 30 MDD = 58 sujets | 30 contrôles (H) + 34 MDD = 64 sujets (données complètes) |
| **Validation croisée (SVM)** | `KFold` (10 plis) | `GroupKFold` (10 plis, groupé par sujet) |
| **Data leakage** | ⚠️ Des époques d'un même sujet pouvaient se retrouver à la fois dans l'ensemble d'entraînement et de test | ✅ Chaque sujet appartient à un seul pli |

> 💡 **Pourquoi le data leakage gonfle les performances ?** Avec `KFold`, le modèle peut avoir "vu" des époques du même sujet lors de l'entraînement, ce qui facilite artificiellement la classification. `GroupKFold` garantit que le modèle est évalué sur des sujets entièrement nouveaux, ce qui reflète mieux la capacité de généralisation réelle.

---

### Résultats — Modèle SVM (accuracy)

| Variante du modèle | Original (avec leakage, 58 sujets) | Corrigé (sans leakage, 64 sujets) |
|---|---|---|
| **SVM de base** (kfold=10) | 0.959 | 0.892 |
| **SVM + SelectPercentile** (20 %, kfold=10) | 0.878 | 0.777 |
| **SVM + SelectKBest** (k=50, kfold=10) | 0.920 | 0.844 |
| **SVM + PCA** (10 composantes, kfold=10) | 0.846 | 0.682 |

---

### Résultats — Modèle EEGNet (accuracy)

| Variante du modèle | Original (Chai, 2025) | Corrigé (Vallée, 2026) |
|---|---|---|
| **EEGNet de base** | ~0.80 | 0.7368 |
| **EEGNet (meilleurs hyperparamètres manuels)** | — | 0.8659 (F1=8, D=4, F2=32) |

> ℹ️ La courbe d'apprentissage du modèle de base révèle un **surapprentissage** (*overfitting*) important : l'accuracy d'entraînement atteint ~0.94 alors que l'accuracy de validation reste instable autour de 0.50–0.74. L'ajustement manuel des hyperparamètres (F1=8, D=4, F2=32) a permis de stabiliser le modèle et d'améliorer significativement la performance de validation à **0.8659**.

---

### Interprétation

- **SVM :** La correction du data leakage entraîne une diminution systématique des performances (de ~3 à ~16 points de pourcentage selon la variante), confirmant que les résultats originaux étaient artificiellement gonflés.
- **EEGNet :** Le modèle de base corrigé (~0.74) performe moins bien que le modèle original (~0.80), mais après ajustement des hyperparamètres, les performances sont comparables (~0.87). La forte variance observée sur la courbe d'apprentissage suggère que le modèle bénéficierait d'un entraînement plus long ou d'une régularisation accrue.

## Références
Chai, W.-X. (2025, 15 juin). brainhack-EEG-depression-ml-dl [Code source]. GitHub. https://github.com/ChaiWeiXuan/brainhack-EEG-depression-ml-dl

Mumtaz, W. (2016, 23 novembre). MDD Patients and Healthy Controls EEG Data (New). figshare. doi:10.6084/m9.figshare.4244171.v2 


## Énoncé d'utilisation de l'IA

### Tâche 1 
- ChatGPT fut utilisé pour m'aider à comprendre certaines parties du code de l'auteur de base
- Copilot GitHub et des Agents fut utilisés pour le débogage de la section Automatisation

### Tâche 2 
- Copilot GitHub fut utilisé pour le débogage suite à la restrucutration du code

### Tâche 3 
- Copilot GitHub fut utilisé pour le débogage suite aux modifications incluant les nouvelles données
- Copilot GitHub fut utilisé pour m'aider à identifier la source du data leakage
- Copilot GitHub fut utiliser pour structurer la documentation en format Markdown

<img src="assets/badge_ia.png" alt="Assisté par l'IA" width="150"/>