# CBIR – Recherche d'images par contenu

## 1. Lancer le projet

Ouvrir CMD dans le dossier du projet :

cd C:\Projects\cbir_project

Lancer l'application :

python app_tkinter.py

Aucun entraînement n'est nécessaire : les index sont déjà enregistrés dans le dossier `features/`.

## 2. Démonstration

1. Sélectionner une image.
2. Choisir le mode Global ou Local (BoVW).
3. Choisir une distance : Euclidean, Manhattan ou Cosine.
4. Lancer la recherche.
5. Observer les images similaires obtenues.

## 3. Explication des méthodes

- Global : compare les caractéristiques générales des images (couleurs, textures, formes et contours).
- Local (ORB + BoVW) : détecte des points d'intérêt et compare les images à partir de leurs descripteurs locaux regroupés en mots visuels.

## 4. Résultats

Dataset : 100 images réparties en 4 catégories.

Meilleur résultat :
- Méthode : Global
- Distance : Manhattan
- MAP : 0,4636

## 5. Réentraînement (uniquement si nécessaire)

Si les fichiers du dossier `features/` sont absents :

python -m src.global_features.index_global

python -m src.local_features.train_bovw

python -m src.local_features.index_bovw

## 6. Présentation rapide

Ce projet est un système de recherche d'images par contenu. Il permet de rechercher des images similaires à partir d'une image fournie par l'utilisateur. J'ai développé deux approches, une globale et une locale basée sur ORB et BoVW, ainsi qu'une interface graphique avec Tkinter.