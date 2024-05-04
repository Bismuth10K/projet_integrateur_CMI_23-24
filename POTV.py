import time

from POTI import POTI
from fonctions import *

rng = np.random.RandomState(1)


class POTV(POTI):
	def __init__(self, path_img_ref, path_vid_tar=None):
		super().__init__(path_img_ref)

		self.new_frames_tar = []
		self.mat_cluster_tar_2 = None
		self.mat_tar_2 = None
		if path_vid_tar is not None:
			start_extraction = time.time()
			self.frames_tar = extract_frames(path_vid_tar)
			end_extraction = time.time()
			print(f"Temps extraction frames : {round(end_extraction - start_extraction, 2)}s")

			self.mat_tar = im2mat(self.frames_tar[0])
			self.mat_cluster_tar = clustering(self.mat_tar, self.model_cluster)
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
			ot_model = ot.da.MappingTransport(mu=1e0, eta=1e-8, bias=True, max_iter=20)
		elif method == "gaussian":
			ot_model = ot.da.MappingTransport(mu=1e0, eta=1e-2, sigma=1, bias=False, max_iter=10)
		else:
			raise Exception("Les quatres choix de modèles sont :\n- 'emd' : EMDTransport \n-'sinkhorn': "
							"SinkhornTransport \n-'linear': Mapping linéaire \n-'gaussian': Mapping Gaussien")

		try:
			ot_model.fit(Xs=self.mat_cluster_tar, Xt=self.mat_cluster_ref)
			self.ot_model = ot_model

			new_img = self.ot_model.transform(Xs=self.mat_cluster_tar)
			img = minmax(mat2im(new_img, self.mat_cluster_tar.shape))
			new_img_created = mat2im(img[self.model_cluster.predict(self.mat_tar), :], self.frames_tar[0].shape)

			self.mat_tar_2 = im2mat(new_img_created)
			self.mat_cluster_tar_2 = clustering(self.mat_tar_2, self.model_cluster)

			# TODO le souci est que la première frame doit sauter car on a pas trié en fonction de la méthode
			ot_model.fit(Xs=self.mat_cluster_tar, Xt=self.mat_cluster_tar_2)
			self.ot_model = ot_model

			print(f"Temps d'entraînement {method} : {round(time.time() - start, 2)}s")
		except Exception as e:
			raise Exception("Pas de cible.")
		return ot_model

	def apply_ot(self, ot_model=None, title_video: list = None):
		start = time.time()

		height, width, layers = self.frames_tar[0].shape
		fourcc = cv2.VideoWriter_fourcc(*'DIVX')
		if ot_model is not None:
			if not isinstance(ot_model, list):
				ot_model = [ot_model]
		else:
			ot_model = [self.ot_model]
		videos = [cv2.VideoWriter(title_video[i] + ".avi", fourcc, 30, (width, height)) for i in range(len(ot_model))]

		print(f"Recolorisation frame 1/{len(self.frames_tar)} - {round(1 / len(self.frames_tar) * 100, 2)}%")
		for count in range(1, len(self.frames_tar)):
			print(
				f"Recolorisation frame {count + 1}/{len(self.frames_tar)} - {round((count + 1) / len(self.frames_tar) * 100, 2)}%")
			mat_img = im2mat(self.frames_tar[count])
			X = clustering(mat_img, self.model_cluster)

			for i in range(len(ot_model)):
				mat_col = color_image(X, ot_model[i], X.shape)
				img_col = mat2im(mat_col[self.model_cluster.predict(mat_img), :], self.frames_tar[count].shape)

				# Mettre à l'échelle de 0-1 à 0-255 pour écrire la vidéo
				videos[i].write(cv2.cvtColor((img_col * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

		nb_sec = round(time.time() - start, 2)
		nb_min = round(nb_sec / 60, 2)
		print(f"Temps de colorisation : {nb_sec} secondes, soit {nb_min} minutes.")
		print("-" * 50)
		cv2.destroyAllWindows()
		for video in videos:
			video.release()

	def create_video(self, title_video: list):
		"""
		Création d'une vidéo (.avi) à partir d'une liste de photos (numpy array)
		Entrées: - frames (numpy list of image's pixels): Images that we want to use for the creation of the video
				 - video_name (str): Name of the video (don't put the .avi)
				 - path (str): path of the video
		"""
		for video_i in range(len(self.new_frames_tar)):
			print(f"__Rendu vidéo n°{video_i}__")
			height, width, layers = self.new_frames_tar[video_i][0].shape
			fourcc = cv2.VideoWriter_fourcc(*'DIVX')
			video = cv2.VideoWriter(title_video[video_i] + ".avi", fourcc, 30, (width, height))
			i = 1
			for frame in self.new_frames_tar[video_i]:
				print(f"\tReconstruction frame {i}")
				i += 1
				video.write(cv2.cvtColor((frame * 255).astype(np.uint8),
										 cv2.COLOR_RGB2BGR))  # Mettre à l'échelle de 0-1 à 0-255 pour écrire la vidéo
			cv2.destroyAllWindows()
			video.release()


if __name__ == '__main__':
	start = time.time()
	img_ref = ['./photos/cathedrale_rouen_monet/La Cathédrale de Rouen.jpg', './photos/picture_city.jpg',
			   './photos/control_game_red_room.jpg']
	vid_tar = './videos/short_city.mp4'

	list_method = ["emd", "sinkhorn", "linear", "gaussian"]

	for img in img_ref:
		potv1 = POTV(img, vid_tar)
		potv1.plot_photos()
		list_model = [potv1.train_ot(method) for method in list_method]
		potv1.apply_ot(list_model, ["rendu_" + img.split("/")[-1][:-4] + "_" + method for method in list_method])
	nb_sec = round(time.time() - start, 2)
	nb_min = round(nb_sec / 60, 2)
	print(f"Temps totale de render : {nb_sec}s, soit {nb_min}min.")
