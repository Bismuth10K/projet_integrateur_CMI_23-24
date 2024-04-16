import numpy as np
from matplotlib import pyplot as plt
import ot
import cv2
import matplotlib.pylab as pl
import sklearn.cluster as skcluster

rng = np.random.RandomState(1)


class POTO:

	def __init__(self, img_ref):
		self.img = img_ref
		self.matrix = self.im2mat(img_ref)

	def im2mat(self, img):
		"""
		Convertie une image en matrice (un pixel par ligne)
		Entrée: - img(matrix): Image que l'on cherche à transformer en matrice
		Sortie: Matrice obtenue après le reshape
		"""
		return img.reshape((img.shape[0] * img.shape[1], img.shape[2]))

	def plot_distribution(self, mat, title):
		"""
		Affiche la distribution des couleurs de la matrice en paramètre
		Entrées : 	- mat (matrice) : Matrice de l'image étudiée
					- title (str) : Titre du graphe
		"""
		plt.scatter(mat[:, 0], mat[:, 2], c=mat)
		plt.axis((0., 1., 0., 1.))
		plt.xlabel('Red')
		plt.ylabel('Blue')
		plt.title(title)
		plt.show()

	def clustering(mat, model_clust):
		"""
		Applique un clustering sur deux matrices d'images (Les valeurs de X doivent être entre 0 et 1).
		Entrées :	- mat (matrice) : Matrice de la première image
					- model_clust (model scikit learn) : Modèle de clustering
		Sortie : - Xs (matrice) : Matrice après le clustering
		"""
		clust1 = model_clust.fit(mat)
		return np.clip(clust1.cluster_centers_, 0, 1)

	def train_tran_opt(Xs, Xt, method):
		"""
		Entraîne un modèle de transport optimal sur les données Xs et Xt.
		Entrées : 	- Xs (matrice) : Matrice à colorer pour l'entraînement (mieux si clustering en amont)
					- Xt (matrice) : Matrice référence pour l'entraînement (mieux si clustering en amont)
					- method (str) : Modèle de transport optimal utilisé
		Sortie : - ot_model (model OT) : Modèle de transport optimal entraîné
		"""
		if method == "emd":
			ot_model = ot.da.EMDTransport()
		elif method == "sinkhorn":
			ot_model = ot.da.SinkhornTransport(reg_e=1e-1)
		elif method == "linear":
			ot_model = ot.da.MappingTransport(
				mu=1e0, eta=1e-8, bias=True, max_iter=20, verbose=True)
		elif method == "gaussian":
			ot_model = ot.da.MappingTransport(
				mu=1e0, eta=1e-2, sigma=1, bias=False, max_iter=10, verbose=True)
		else:
			print("Les quatres choix de modèles sont : \n\
				-'emd' : EMDTransport \n\
				-'sinkhorn': SinkhornTransport \n\
				-'linear': Mapping linéaire \n\
				-'gaussian': Mapping Gaussien")
			return None

		ot_model.fit(Xs=Xs, Xt=Xt)
		return ot_model
