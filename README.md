# CBIR – Recherche d'images par contenu

Système de **Content-Based Image Retrieval** : à partir d'une image requête, l'application retrouve les images les plus similaires d'un dataset, via une interface graphique Tkinter.

Le projet implémente et compare deux approches complémentaires :

- une approche **globale** basée sur des descripteurs de couleur, texture, forme et contours ;
- une approche **locale** basée sur ORB et la technique **BoVW** (Bag of Visual Words).

## 1. Fonctionnalités

- Recherche d'images similaires à partir d'une image fournie par l'utilisateur.
- Deux modes de recherche : Global et Local (ORB + BoVW).
- Trois mesures de comparaison : Euclidean, Manhattan et Cosine.
- Interface graphique Tkinter avec affichage des images résultats.
- Évaluation quantitative des performances (Precision, Recall, F1, MAP).

## 2. Approches

- **Global** : compare les caractéristiques générales des images (histogrammes de couleur RGB/HSV, LBP, moments de Hu, HOG, moments de couleur et histogramme de contours).
- **Local (ORB + BoVW)** : détecte des points d'intérêt avec ORB, regroupe leurs descripteurs en mots visuels par K-Means, puis compare les images à partir de leur histogramme de mots visuels.

## 3. Technologies

- **Python**
- **OpenCV** (ORB, traitement d'image)
- **scikit-image** (LBP, HOG)
- **scikit-learn** (K-Means)
- **NumPy / SciPy** (calcul vectoriel et distances)
- **Pillow** (affichage des images)
- **Tkinter** (interface graphique)

## 4. Structure du projet

```text
cbir_project/
├── app_tkinter.py              # Application graphique
├── data/                       # Dataset (non versionné)
├── features/                   # Index et vocabulaire déjà calculés
├── results/                    # Rapport d'évaluation
├── src/
│   ├── extract.py              # Extraction des descripteurs
│   ├── global_features/        # Indexation et recherche globale
│   ├── local_features/         # ORB, BoVW, vocabulaire et recherche locale
│   └── evaluation/             # Métriques et évaluation
├── requirements.txt
└── README.md
```

## 5. Installation

Prérequis : Python 3.10 ou plus récent.

```bash
pip install -r requirements.txt
```

## 6. Lancer l'application

```bash
cd C:\Projects\cbir_project
python app_tkinter.py
```

Le dossier `data/` n'est pas versionné sur GitHub : après un clone, il faut donc ajouter vos propres images (voir section 8) puis relancer l'indexation (voir section 10). Sans cela, l'application se lance mais les chemins enregistrés dans les index ne correspondent à aucune image.

Les index déjà présents dans `features/` ne sont valables que pour le dataset utilisé lors de leur création : si les images de `data/` changent, il faut les recalculer.

## 7. Démonstration

1. Sélectionner une image requête.
2. Choisir le mode Global ou Local (BoVW).
3. Choisir une distance : Euclidean, Manhattan ou Cosine.
4. Lancer la recherche.
5. Observer les images similaires obtenues.

## 8. Dataset et personnalisation

Seuls les fichiers de données (les images du dossier `data/`) sont ignorés par Git : **le dossier `data/` n'est donc pas versionné sur GitHub** et n'est pas présent après un `git clone`. Le reste du projet, notamment les index déjà calculés dans `features/`, est bien versionné.

Le projet n'est pas limité aux images fournies ici : le code peut être adapté avec les sous-dossiers de votre choix pour fonctionner avec n'importe quel type d'images. Chaque sous-dossier de `data/` représente simplement une catégorie (classe) d'images :

```text
data/
├── chats/
│   ├── image1.jpg
│   └── image2.png
├── chiens/
│   └── image3.jpeg
└── voitures/
    └── image4.bmp
```

Le nombre de sous-dossiers, leurs noms et le nombre d'images par dossier sont libres : le code parcourt automatiquement tout le contenu de `data/` de façon récursive. Les formats d'image acceptés sont `.jpg`, `.jpeg`, `.png` et `.bmp`.

Pour utiliser vos propres images :

1. Créez le dossier `data/` s'il n'existe pas, puis placez-y vos images organisées dans les sous-dossiers de votre choix (un sous-dossier par catégorie).
2. Relancez l'indexation (voir section 10) afin de recalculer les index sur le nouveau dataset.
3. Lancez l'application et effectuez vos recherches.

## 9. Résultats

Résultats obtenus sur le dataset de démonstration (100 images, 4 catégories), qui n'est pas versionné sur GitHub.

Meilleur résultat :

- Méthode : Global
- Distance : Manhattan
- MAP : 0,4636

Le rapport complet est disponible dans `results/evaluation_report.csv`.

## 10. Indexation / réentraînement

À relancer après un clone (le dossier `data/` n'étant pas versionné) ou dès que les images de `data/` changent :

```bash
python -m src.global_features.index_global
python -m src.local_features.train_bovw
python -m src.local_features.index_bovw
```

## 11. Évaluation

Pour recalculer les métriques sur le dataset courant :

```bash
python -m src.evaluation.runner
```

Le rapport est enregistré dans `results/evaluation_report.csv`.
