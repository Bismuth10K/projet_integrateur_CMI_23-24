import time

from fonctions import *


class POTI:
	"""
	Cette classe permet de recoloriser une image cible grâce aux couleurs d'une image de référence.
	Ce script utilise la librairie POT (Python Optimal Transport) pour recoloriser les images.
	"""

	def __init__(self, path_img_ref: str, path_img_tar: str = None):
		"""
		Constructeur de la classe.

		Parameters
		----------
		path_img_ref (str) : Chemin vers l'image de référence
		path_img_tar (str) : Chemin vers l'image cible
		"""
		start = time.time()

		self.model_cluster = MiniBatchKMeans(n_clusters=1000, init_size=3000, random_state=2)
		self.ot_model = None
		self.img_ref = plt.imread(path_img_ref).astype(np.float64) / 256
		self.mat_ref = im2mat(self.img_ref)
		self.mat_cluster_ref = clustering(self.mat_ref, self.model_cluster)
		end_cluster_1 = time.time()
		print(f"Temps de clustering image référence : {round(end_cluster_1 - start, 2)}s")

		if path_img_tar is not None:
			self.img_tar = plt.imread(path_img_tar).astype(np.float64) / 256
			self.mat_tar = im2mat(self.img_tar)
			self.mat_cluster_tar = clustering(self.mat_tar, self.model_cluster)
			end_cluster_2 = time.time()
			print(f"Temps de clustering image cible : {round(end_cluster_2 - end_cluster_1, 2)}s")
		else:
			self.img_tar = None
			self.mat_tar = None
			self.mat_cluster_tar = None

	def set_target(self, path_img_tar: str):
		"""
		Setter de l'image cible.
		Cette fonction est une version simple (d'un point de vue utilisateur) où on applique le clustering sur l'image.
		Cette fonction peut donc prendre du temps.

		Parameters
		----------
		path_img_tar (str) : Chemin vers l'image cible
		"""
		self.img_tar = plt.imread(path_img_tar).astype(np.float64) / 256
		self.mat_tar = im2mat(self.img_tar)
		self.mat_cluster_tar = clustering(self.mat_tar, self.model_cluster)

	def set_target_all(self, img: np.ndarray, mat: np.ndarray, mat_cluster: np.ndarray, model_cluster):
		"""
		Setter de l'image cible.
		Cette fonction est une version plus complexe où nous recevons toutes les infos nécessaires concernant l'image cible.
		Cette fonction est donc plus rapide que celle du dessus, car le clustering a déjà été appliqué.

		Elle est utile dans le cas d'une automatisation de render d'une série afin de limiter les coûts de cluster.

		Si la fonction demande aussi le modèle de cluster, c'est parce que nous avons besoin d'un cluster entraîné sur
		l'image cible pour appliquer le Transport Optimal. Sinon le résultat est peu appréciable.

		Parameters
		----------
		img : Image cible importée par matplotlib
		mat : Matrice de l'image cible
		mat_cluster : Matrice du clustering appliqué sur cette image grâce à model_cluster
		model_cluster : Modèle de clustering qui a servi à cluster l'image
		"""
		self.img_tar = img
		self.mat_tar = mat
		self.mat_cluster_tar = mat_cluster
		self.model_cluster = model_cluster

	def get_ref_all(self):
		"""
		Renvoie toutes les informations de l'image de référence.
		Est utile dans le cas d'une automatisation de séries.

		Returns
		-------
		Tableau contenant dans l'ordre : L'image, sa matrice, la matrice du cluster, le modèle de cluster.
		"""
		return [self.img_ref, self.mat_ref, self.mat_cluster_ref, self.model_cluster]

	def plot_distributions(self):
		"""
		Affichage des distributions des couleurs des images.
		S'il n'y a pas l'image cible, on affichera seulement celle de l'image référence.
		"""
		if self.mat_cluster_tar is not None:
			fig, axs = plt.subplots(2)
			fig.tight_layout(pad=2.0)
			axs[1] = plot_distribution(axs[1], self.mat_cluster_tar, "Distribution de couleur de l'image cible")
		else:
			figs, axs = plt.subplots(1)
		axs[0] = plot_distribution(axs[0], self.mat_cluster_ref, "Distribution de couleur de l'image de référence")
		plt.show()

	def plot_photos(self):
		"""
		Affichage des images.
		S'il n'y a pas l'image cible, on affichera seulement l'image référence.
		"""
		if self.img_tar is not None:
			fig, axs = plt.subplots(2)
			fig.tight_layout(pad=2.0)
			axs[1].imshow(self.img_tar)
			axs[1].set_axis_off()
			axs[1].set_title("Image cible")
		else:
			figs, axs = plt.subplots(1)
		axs[0].imshow(self.img_ref)
		axs[0].set_axis_off()
		axs[0].set_title("Image de référence")
		plt.show()

	def train_ot(self, method: str = "emd"):
		"""
		Applique le transfert optimal des couleurs de l'image de référence sur les couleurs de l'image cible en
		fonction de l'algorithme choisi par l'utilisateur.

		Parameters
		----------
		method (str) : Modèle de transport optimal utilisé (EMD par défaut)

		Returns
		-------
		Le modèle entraîné.
		"""
		start = time.time()
		if method == "emd":
			ot_model = ot.da.EMDTransport()
		elif method == "sinkhorn":
			ot_model = ot.da.SinkhornTransport(reg_e=1e-1)
		elif method == "linear":
			ot_model = ot.da.MappingTransport(mu=1e0, eta=1e-8, bias=True, max_iter=20)
		elif method == "gaussian":
			ot_model = ot.da.MappingTransport(mu=1e0, eta=1e-2, sigma=1, bias=False, max_iter=10)
		else:
			raise Exception("Les quatre choix de modèles sont :\n- 'emd' : EMDTransport \n-'sinkhorn': "
							"SinkhornTransport \n-'linear': Mapping linéaire \n-'gaussian': Mapping Gaussien")

		try:
			ot_model.fit(Xs=self.mat_cluster_tar, Xt=self.mat_cluster_ref)
			self.ot_model = ot_model

			print(f"Temps d'entraînement : {round(time.time() - start, 2)}s")
		except Exception:
			raise Exception("Pas de cible.")

	def apply_ot(self):
		"""
		Applique le modèle sur l'image cible.

		Returns
		-------
		L'image cible recolorisée avec les couleurs de l'image de référence.
		"""
		start = time.time()
		new_img = self.ot_model.transform(Xs=self.mat_cluster_tar)
		img = minmax(mat2im(new_img, self.mat_cluster_tar.shape))
		img_col = mat2im(img[self.model_cluster.predict(self.mat_tar), :], self.img_tar.shape)
		print(f"Temps de colorisation : {round(time.time() - start, 2)}s")
		return img_col
