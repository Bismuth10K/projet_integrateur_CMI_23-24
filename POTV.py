import time

from POTI import POTI
from fonctions import *

rng = np.random.RandomState(1)


class POTV(POTI):
	def __init__(self, path_img_ref, vid_tar=None):
		super().__init__(path_img_ref)

		if vid_tar is not None:
			start_extraction = time.time()
			self.frames_tar = extract_frames(vid_tar)
			end_extraction = time.time()
			print(f"Temps extraction frames : {round(end_extraction - start_extraction, 2)}s")

			self.mat_tar = im2mat(self.frames_tar[0])
			self.mat_clstr_tar = clustering(self.mat_tar, self.model_cluster)
			end_clust_1 = time.time()
			print(f"Temps de clustering frame 1 : {round(end_clust_1 - end_extraction, 2)}s")

	def plot_photos(self):
		if self.frames_tar is not None:
			fig, axs = plt.subplots(2)
			fig.tight_layout(pad=2.0)
			axs[1].imshow(self.frames_tar[0])
			axs[1].set_axis_off()
			axs[1].set_title("Image cible")
		else:
			figs, axs = plt.subplots(1)
		axs[0].imshow(self.img_ref)
		axs[0].set_axis_off()
		axs[0].set_title("Image de référence")
		plt.show()

	def train_ot(self, method="emd"):
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
			ot_model.fit(Xs=self.mat_clstr_tar, Xt=self.mat_cluster_ref)
			self.ot_model = ot_model

			new_img = self.ot_model.transform(Xs=self.mat_clstr_tar)
			img = minmax(mat2im(new_img, self.mat_clstr_tar.shape))
			self.frames_tar[0] = mat2im(img[self.model_cluster.predict(self.mat_tar), :], self.frames_tar[0].shape)

			self.mat_tar = im2mat(self.frames_tar[0])
			self.mat_clstr_tar = clustering(self.mat_tar, self.model_cluster)

			print(f"Temps d'entraînement : {round(time.time() - start, 2)}s")
		except Exception as e:
			raise Exception("Pas de cible.")
		self.plot_photos()
		self.plot_distribution()

	def apply_ot(self):
		start = time.time()
		for count in range(1, len(self.frames_tar)):
			mat_img = im2mat(self.frames_tar[count])
			X = clustering(mat_img, self.model_cluster)
			mat_col = color_image(X, self.ot_model, X.shape)
			img_col = mat2im(mat_col[self.model_cluster.predict(mat_img), :], self.frames_tar[count].shape)
			self.frames_tar[count] = img_col
			print(f"Recolorisation frame {count + 1}/{len(self.frames_tar)} - {round((count + 1) / len(self.frames_tar) * 100, 2)}%")

		print(f"Temps de colorisation : {round(time.time() - start, 2)}s")

	def create_video(self):
		"""
		Création d'une vidéo (.avi) à partir d'une liste de photos (nupy array)
		Entrées: - frames (numpy list of image's pixels): Images that we want to use for the creation of the video
				 - video_name (str): Name of the video (don't put the .avi)
				 - path (str): path of the video

		"""
		height, width, layers = self.frames_tar[0].shape
		fourcc = cv2.VideoWriter_fourcc(*'DIVX')
		video = cv2.VideoWriter("rendu_video.avi", fourcc, 30, (width, height))

		i = 1
		for frame in self.frames_tar:
			print(f"Reconstruction frame {i}")
			i += 1
			video.write((frame * 255).astype(np.uint8))  # Mettre à l'échelle de 0-1 à 0-255 pour écrire la vidéo

		cv2.destroyAllWindows()
		video.release()


if __name__ == '__main__':
	start = time.time()
	img_ref = './photos/picture_city.jpg'
	vid_tar = './videos/short_city.mp4'

	potv1 = POTV(img_ref, vid_tar)
	potv1.plot_photos()
	potv1.plot_distribution()
	potv1.train_ot()
	potv1.apply_ot()
	potv1.create_video()
	nb_sec = round(time.time() - start, 2)
	nb_min = round(nb_sec / 60, 2)
	print(f"Temps totale de render : {nb_sec}s, soit {nb_min}min.")
