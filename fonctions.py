import cv2
import numpy as np
import ot
from matplotlib import pyplot as plt
from sklearn.cluster import MiniBatchKMeans

rng = np.random.RandomState(1)


def im2mat(img):
	"""
	Convertie une image en matrice (un pixel par ligne)
	Entrée : - img(matrix): Image que l'on cherche à transformer en matrice
	Sortie : Matrice obtenue après le reshape
	"""
	return img.reshape((img.shape[0] * img.shape[1], img.shape[2]))


def mat2im(X, shape):
	"""
	Convertie une matrice en image
	Entrées: - X(matrice): Matrice que l'on veut convertir en image
			 - shape(list): Dimensions de l'image attendue
	Sortie: Image obtenue
	"""
	return X.reshape(shape)


def minmax(img):
	"""
	Limite les valeurs de l'image entre 0 et 1.
	Entrée: - img(matrice): Image à modifier
	Sortie: - np.clip(img,0,1)(matrice): Image avec des valeurs uniquement entre 0 et 1
	"""
	return np.clip(img, 0, 1)


def import_image(path):
	"""
	Importe une image et récupère sa matrice
	Entrée: - path(str): Chemin de l'image
	Sorties: - img(matrix): Image obtenue
			 - mat(matrix): Matrice de l'image
	"""
	img = plt.imread(path).astype(np.float64) / 256
	mat = im2mat(img)
	return img, mat


def print_image(ax, img, title):
	"""
	Affiche l'image dans l'API utilisée
	Entrées: - img(matrix): Image à afficher
			 - title(str): Titre de l'image
	"""
	ax.imshow(img)
	ax.axis('off')
	ax.title.set_text(title)
	return ax


def extract_frames(path):
	"""
	Récupère les images d'une vidéo
	Entrée: - path(str): Chemin pour accéder à la vidéo
	Sortie: - frames(list): Liste des images qui composent la vidéo
	"""
	cap = cv2.VideoCapture(path)
	frames = []

	while True:
		ret, frame = cap.read()
		if not ret:
			break
		frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # récupère les images en RGB
		frames.append(frame_rgb.astype(np.float64) / 256)

	cap.release()  # libère cap

	return frames


def clustering(X, model_clust=MiniBatchKMeans(n_clusters=1000, init_size=3000, random_state=2)):
	"""
	Applique un clustering sur deux matrices d'images (Les valeurs de X doivent être entre 0 et 1).
	Entrées: - X1(matrice): Matrice de la première image
			 - model_cluster(model scikit learn): Modèle de clustering
	Sortie: - Xs(matrice): Matrice après le clustering
	"""
	idx1 = rng.randint(X.shape[0], size=(500,))
	clust1 = model_clust.fit(X)
	Xs = np.clip(clust1.cluster_centers_, 0, 1)
	return Xs


def plot_distribution(ax, X, title):
	"""
	Affiche la distribution des couleurs de la matrice en paramètre
	Entrées:
				- ax (subplots) : Objet plot de matplotlib
				- X(matrice): Matrice de l'image étudiée
				- title(str): Titre du graphe
	"""
	ax.scatter(X[:, 0], X[:, 2], c=X)
	ax.axis([0, 1, 0, 1])
	ax.set(xlabel='Red', ylabel='Blue')
	ax.set_title(title)
	return ax


def tran_opt(Xs, Xt, X, img_shape, method="emd"):
	"""
	Applique le transfert optimal sur une image X en fonction de l'algorithme choisi par l'utilisateur
	Entrées: - Xs(matrice): Matrice après clustering de l'image à colorer
			 - Xt(matrice): Matrice après clustering de l'image référence
			 - X(matrice): Matrice de l'image à colorer
			 - img_shape(shape): Dimension de l'image à colorer
			 - method(str): Modèle de transport optimal utilisé (EMD par défaut)
	Sortie: - Image recolorisée
	"""
	if method == "emd":
		ot_model = ot.da.EMDTransport()
	elif method == "sinkhorn":
		ot_model = ot.da.SinkhornTransport(reg_e=1e-1)
	elif method == "linear":
		ot_model = ot.da.MappingTransport(mu=1e0, eta=1e-8, bias=True, max_iter=20, verbose=True)
	elif method == "gaussian":
		ot_mapping_gaussian = ot.da.MappingTransport(mu=1e0, eta=1e-2, sigma=1, bias=False, max_iter=10, verbose=True)
	else:
		print(
			"Les quatres choix de modèles sont : \n- 'emd' : EMDTransport \n-'sinkhorn': SinkhornTransport \n-'linear': Mapping linéaire \n-'gaussian': Mapping Gaussien")
		return None

	ot_model.fit(Xs=Xs, Xt=Xt)
	transp_Xs = ot_model.transform(Xs=X)
	img = minmax(mat2im(transp_Xs, img_shape))
	return img


def color_image(X, ot_model, img_shape):
	"""
	Recolorise une image X à partir d'un modèle d'optimal transport entraîné
	Entrées: - X (matrice): Image à recolorer
			 - ot_model (model OT): Modèle de transport optimal entraîné
			 - img_shape (vector): Dimensions de l'image X
	Sortie: - img (matrice): Image recolorée
	"""
	transp_Xs = ot_model.transform(Xs=X)
	img = minmax(mat2im(transp_Xs, img_shape))
	return img
