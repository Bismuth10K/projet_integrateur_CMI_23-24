# Projet intégrateur des CMI (Année 23-24)

Voici le projet intégrateur des CMI de l'UBS pour l'année scolaire 2023-2024.
Nous devons appliquer une méthode permettant de changer les couleurs d'une image ou d'une vidéo pour les couleurs d'une image de référence, et ce graĉe la bibliothèque [POT](https://github.com/PythonOT/POT).

# Exécution
Cette section devra encore être développée à l'avenir lorsque le code sera plus avancé.

## Dépendances
Plusieurs dépendances sont nécessaires, voici la commande `pip` :
```bash
$ pip install opencv-python pot matplotlib scikit-learn numpy
```

> Vous pouvez les placer dans un environnement virtuel en utilisant les commandes suivantes
> (*À faire avant d'installer les dépendances*) :

### Linux
```bash
$ python -m venv .venv
$ source .venv/bin/activate
```

### Windows
*À venir*

## Lancement
Le repo contient plusieurs scripts qui ne sont pas encore rangés.

- `notebook.ipynb` est le fichier Jupyter de base où nous avons fait les premiers tests.
Sachant qu'il marche, vous pouvez l'exécuter afin de comprendre le fonctionnement du POT.
- `POTO.py` est le fichier contenant la classe POTO afin de transformer une image.
Le fichier doit encore être commenté, mais vous trouverez un exemple de code dans le bas du script, vous pouvez modifier les chemins des images source et cible.
- `automatisation_series.py` permet d'automatiser un *render* sur toutes les images d'un dossier.
Vous pouvez, là aussi, modifier le chemin contenant des images.
- `fonctions.py` contient plusieurs fonctions utiles au fonctionnement de `POTO.py`