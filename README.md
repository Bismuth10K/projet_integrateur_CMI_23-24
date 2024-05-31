# Projet intégrateur des CMI (Année 23-24)

Voici le projet intégrateur des CMI de l'UBS pour l'année scolaire 2023-2024.
Nous devons appliquer une méthode permettant de changer les couleurs d'une image ou d'une vidéo pour les couleurs d'une image de référence, et ce graĉe la bibliothèque [POT](https://github.com/PythonOT/POT).

La classe POTI permet de recoloriser des images, POTV des vidéos.

## Dépendances
Plusieurs dépendances sont nécessaires, voici la commande `pip` :

```bash
$ pip install opencv-python pot matplotlib scikit-learn numpy tqdm moviepy tensorflow
```

> Vous pouvez éventuellement les installer dans un environnement virtuel en utilisant les commandes suivantes.
> (*À faire avant d'installer les dépendances*) :

### Linux
```bash
$ python -m venv .venv
$ source .venv/bin/activate
```

## Lancement
Le repo contient plusieurs scripts.

- [`notebook.ipynb`](./notebook.ipynb) : Premiers programmes pour appréhender l'outil.
- [`fonctions.py`](./fonctions.py) : Ensemble de fonctions utiles au traitement du transport optimal et des images.
- [`POTI.py`](./POTI.py) (*Python Optimal Transport for Images*) : Classe permettant d'appliquer un transport optimal d'une image vers une autre.
- [`POTV.py`](./POTV.py) (*Python Optimal Transport for Videos*) : Classe permettant d'appliquer un transport optimal d'une image vers une vidéo. 
- [`automatisation_series.py`](./automatisation_series.py) : Expérience annexe permettant de coloriser plusieurs images entre elles.
- [`style_transfert.ipynb`](./style_transfer.ipynb) : Notebook pour les transferts de caractéristiques par réseaux de convolution.
- [`videoson.py`](./videoson.py) : Pour récupérer le son et l'ajouter aux nouvelles vidéos.

Vous pouvez lancer le script [`runner.py`](./runner.py) et taper le numéro de ce que vous souhaitez exécuter.
Sinon, vous pouvez vous inspirer des morceaux de codes dans ce script et lancer avec vos propres images/vidéos.

### *Disclaimer*
- Pour les images, convertissez-les en .jpg.
Cela ne marche pas pour les .png.
- En fonction de la taille de votre vidéo et de votre éditeur de code, vous aurez peut-être besoin d'augmenter la taille de la *Heap*.
Une recherche d'optimisation est en cours pour ce problème.
Cela est nécessaire si vous obtenez l'erreur suivante (ou toute autre erreur concernant un manque d'espace) :
````
MemoryError: Unable to allocate <space> for an array with shape (<x>, <y>, <z>) and data type float64
````
