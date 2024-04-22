import time

import matplotlib.image
from PIL import Image
from fonctions import *

rng = np.random.RandomState(1)


class POTO:
	def __init__(self, img_ref, img_tar=None):
		start = time.time()

		self.model_clust = MiniBatchKMeans(n_clusters=1000, init_size=3000, random_state=2)
		self.ot_model = None
		self.img_ref = img_ref
		self.mat_ref = im2mat(img_ref)
		self.mat_clstr_ref = clustering(self.mat_ref, self.model_clust)
		end_clust_1 = time.time()
		print(f"Temps de clustering image référence : {round(end_clust_1 - start, 2)}s")

		if img_tar is not None:
			self.img_tar = img_tar
			self.mat_tar = im2mat(img_tar)
			self.mat_clstr_tar = clustering(self.mat_tar, self.model_clust)
			end_clust_2 = time.time()
			print(f"Temps de clustering image cible : {round(end_clust_2 - end_clust_1, 2)}s")
		else:
			self.img_tar = None
			self.mat_tar = None
			self.mat_clstr_tar = None

	def set_target(self, img_tar):
		self.img_tar = img_tar
		self.mat_tar = im2mat(img_tar)
		self.mat_clstr_tar = clustering(self.mat_tar, self.model_clust)

	def set_target_all(self, img, mat, mat_clstr, model_clust):
		self.img_tar = img
		self.mat_tar = mat
		self.mat_clstr_tar = mat_clstr
		self.model_clust = model_clust

	def get_ref_all(self):
		return [self.img_ref, self.mat_ref, self.mat_clstr_ref, self.model_clust]

	def plot_distribution(self):
		if self.mat_clstr_tar is not None:
			fig, axs = plt.subplots(2)
			fig.tight_layout(pad=2.0)
			axs[1] = plot_distribution(axs[1], self.mat_clstr_tar, "Distribution de couleur de l'image cible")
		else:
			figs, axs = plt.subplots(1)
		axs[0] = plot_distribution(axs[0], self.mat_clstr_ref, "Distribution de couleur de l'image de référence")
		plt.show()

	def plot_photos(self):
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

			print(f"Temps d'entraînement : {round(time.time() - start, 2)}s")
		except Exception as e:
			raise Exception("Pas de cible.")

	def apply_ot(self):
		start = time.time()
		new_img = self.ot_model.transform(Xs=self.mat_clstr_tar)
		img = minmax(mat2im(new_img, self.mat_clstr_tar.shape))
		img_col = mat2im(img[self.model_clust.predict(self.mat_tar), :], self.img_tar.shape)
		# plt.imshow(img_col)
		# plt.show()
		print(f"Temps de colorisation : {round(time.time() - start, 2)}s")
		return img_col


if __name__ == '__main__':
	path_ref = './photos/control_game_red_room.jpg'
	img_ref, mat_ref = import_image(path_ref)

	path_tar = './photos/picture_city.jpg'
	img_tar, mat_tar = import_image(path_tar)

	poto1 = POTO(img_ref, img_tar)
	poto1.plot_photos()
	poto1.plot_distribution()
	poto1.train_ot()
	matplotlib.image.imsave('test_normal.png', poto1.apply_ot())
