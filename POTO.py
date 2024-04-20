import time

import matplotlib.pyplot as plt

from fonctions import *

rng = np.random.RandomState(1)


class POTO:
	def __init__(self, img_ref, img_tar=None):
		start = time.time()

		self.ot_model = None
		self.img_ref = img_ref
		self.mat_ref = im2mat(img_ref)
		self.mat_clstr_ref = clustering(self.mat_ref)
		end_clust_1 = time.time()

		if img_tar is not None:
			self.img_tar = img_tar
			self.mat_tar = im2mat(img_tar)
			self.mat_clstr_tar = clustering(self.mat_tar)
		else:
			self.img_tar = None
			self.mat_tar = None
			self.mat_clstr_tar = None
		end_clust_2 = time.time()

		print("Temps de clustering 1: ", end_clust_1 - start)
		print("Temps de clustering 2: ", end_clust_2 - end_clust_1)

	def set_target(self, img_tar):
		self.img_tar = img_tar
		self.mat_tar = im2mat(img_tar)
		self.mat_clstr_tar = clustering(self.mat_tar)

	def plot_distribution(self):
		if self.mat_clstr_tar is not None:
			figs, axs = plt.subplots(2, 1)
			axs[0] = plot_distribution(axs[0], self.mat_clstr_ref, "Distribution de couleur de l'image de référence")
			axs[1] = plot_distribution(axs[1], self.mat_clstr_tar, "Distribution de couleur de l'image cible")
			plt.show()
		else:
			figs, axs = plt.subplots(1, 1)
			axs[0] = plot_distribution(axs[0], self.mat_clstr_ref, "Distribution de couleur de l'image de référence")
			plt.show()

	def plot_photos(self):
		if self.img_tar is not None:
			figs, axs = plt.subplots(2, 1)
			axs[1].imshow(self.img_tar)
			axs[1].set_axis_off()
			axs[1].set_title("Distribution de couleur de l'image cible")
		else:
			figs, axs = plt.subplots(1, 1)
		axs[0].imshow(self.img_ref)
		axs[0].set_axis_off()
		axs[0].set_title("Distribution de couleur de l'image de référence")
		plt.show()

	def train_ot(self, method="emd"):
		"""
		Applique le transfert optimal sur une image X en fonction de l'algorithme choisi par l'utilisateur
		Entrées : 	- method(str) : Modèle de transport optimal utilisé (EMD par défaut)
		Sortie :	- modèle entraîné
		"""
		start = time.time()
		if method == "emd":
			ot_model = ot.da.EMDTransport()
		elif method == "sinkhorn":
			ot_model = ot.da.SinkhornTransport(reg_e=1e-1)
		elif method == "linear":
			ot_model = ot.da.MappingTransport(mu=1e0, eta=1e-8, bias=True, max_iter=20, verbose=True)
		elif method == "gaussian":
			ot_model = ot.da.MappingTransport(mu=1e0, eta=1e-2, sigma=1, bias=False, max_iter=10, verbose=True)
		else:
			raise Exception("Les quatres choix de modèles sont :\n- 'emd' : EMDTransport \n-'sinkhorn': "
							"SinkhornTransport \n-'linear': Mapping linéaire \n-'gaussian': Mapping Gaussien")

		try:
			ot_model.fit(Xs=self.mat_clstr_tar, Xt=self.mat_clstr_ref)
			self.ot_model = ot_model

			end = time.time()
			print("Temps de color: ", end - start)

			return ot_model
		except Exception as e:
			raise Exception("Pas de cible.")

	def apply_ot(self):
		new_img = self.ot_model.transform(Xs=self.img_tar)
		img = minmax(mat2im(new_img, self.img_tar.shape))
		plt.imshow(img)
		return img


if __name__ == '__main__':
	path_ref = './picture_city.jpg'
	img_ref, mat_ref = import_image(path_ref)

	path_tar = './control_game_red_room.jpg'
	img_tar, mat_tar = import_image(path_tar)

	poto1 = POTO(img_ref, img_tar)
	poto1.plot_photos()
	# poto1.plot_distribution()
	poto1.train_ot()
	poto1.apply_ot()
