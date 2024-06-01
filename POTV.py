from tqdm import tqdm #import de la barre de progression
import time

from POTI import POTI
from fonctions import *
from videoson import VideoSon



class POTV(POTI):
	"""
	Cette classe permet de recoloriser une image cible grâce aux couleurs d'une image de référence.
	Ce script utilise la librairie POT (Python Optimal Transport) pour recoloriser les images.
	"""

	def __init__(self, path_img_ref: str, path_vid_tar: str = None):
		"""
		Constructeur de la classe.

		Parameters
		----------
		path_img_ref (str) : Chemin vers l'image de référence
		path_vid_tar (str) : Chemin vers la vidéo cible
		"""
		super().__init__(path_img_ref)
		self.path_vid_tar = path_vid_tar

		self.frames_tar = []
		if path_vid_tar is not None:
			start_extraction = time.time()
			self.frames_tar = extract_frames(path_vid_tar)
			end_extraction = time.time()
			print(
				f"Temps extraction des {len(self.frames_tar)} frames : {round(end_extraction - start_extraction, 2)}s.")

			self.mat_tar = im2mat(self.frames_tar[0])
			self.mat_cluster_tar = clustering(self.mat_tar, self.model_cluster)
			end_clust_1 = time.time()
			print(f"Temps de clustering frame 1 : {round(end_clust_1 - end_extraction, 2)}s")

	def set_reference(self, path_img_ref: str):
		"""
		Pour changer l'image de référence.
		Peut être utile à des fins d'optimisation si on veut tester avec plusieurs images références différentes.

		Parameters
		----------
		path_img_ref (str) : Chemin vers l'image de référence.
		"""
		start = time.time()
		self.img_ref = plt.imread(path_img_ref).astype(np.float64) / 256
		self.mat_ref = im2mat(self.img_ref)
		self.mat_cluster_ref = clustering(self.mat_ref, self.model_cluster)
		end_cluster_1 = time.time()
		print(f"Temps de clustering nouvelle référence : {round(end_cluster_1 - start, 2)}s")

	def plot_photos(self):
		"""
		Affichage des images.
		S'il n'y a pas l'image cible, on affichera seulement l'image référence.
		"""
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
		"""
		Applique le transfert optimal des couleurs de l'image de référence sur les couleurs de la vidéo cible en
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
			raise Exception("Les quatres choix de modèles sont :\n- 'emd' : EMDTransport \n-'sinkhorn': "
							"SinkhornTransport \n-'linear': Mapping linéaire \n-'gaussian': Mapping Gaussien")

		try:
			ot_model.fit(Xs=self.mat_cluster_tar, Xt=self.mat_cluster_ref)

			frame0_recolored = ot_model.transform(Xs=self.mat_cluster_tar)
			frame0_reconstructed = minmax(mat2im(frame0_recolored, self.mat_cluster_tar.shape))
			frame0_image = mat2im(frame0_reconstructed[self.model_cluster.predict(self.mat_tar), :],
								  self.frames_tar[0].shape)

			mat_tar_2 = im2mat(frame0_image)
			mat_cluster_tar_2 = clustering(mat_tar_2, self.model_cluster)

			ot_model.fit(Xs=self.mat_cluster_tar, Xt=mat_cluster_tar_2)
			self.ot_model = ot_model

			print(f"Temps d'entraînement {method} : {round(time.time() - start, 2)}s")
		except Exception as e:
			raise Exception("Pas de cible.")
		return ot_model

	def apply_ot(self, ot_model=None, title_video: list = None):
		"""
		Applique le ou les modèle(s) sur la vidéo cible.

		Génère autant de vidéos qu'il y a de modèles.
		Un paramètre pourra être créé pour définir le chemin de rendu.

		Parameters
		----------
		ot_model : Peut être soit une liste de modèles, soit un modèle seul, soit None.
					Si None, le modèle appliqué sera l'attribut self.ot_model
		title_video : Liste de noms pour les vidéos
		"""
		start = time.time()

		height, width, layers = self.frames_tar[0].shape
		fourcc = cv2.VideoWriter_fourcc(*'DIVX')
		if ot_model is not None:
			if not isinstance(ot_model, list):
				ot_model = [ot_model]
		else:
			ot_model = [self.ot_model]
		videos = [cv2.VideoWriter(title_video[i] + ".mkv", fourcc, 30, (width, height)) for i in range(len(ot_model))]

		for count in tqdm(range(len(self.frames_tar)), desc="Recolorisation frames"):
			# print(f"Recolorisation frame {count + 1}/{len(self.frames_tar)} - {round((count + 1) / len(self.frames_tar) * 100, 2)}%")
			mat_img = im2mat(self.frames_tar[count])
			X = clustering(mat_img, self.model_cluster)

			for i in range(len(ot_model)):
				img_col = color_image(self.frames_tar[count], X, ot_model[i], self.model_cluster)

				# Mettre à l'échelle de 0-1 à 0-255 pour écrire la vidéo
				videos[i].write(cv2.cvtColor((img_col * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

		nb_sec = round(time.time() - start, 2)
		nb_min = round(nb_sec / 60, 2)
		print(f"Temps de colorisation : {nb_sec} secondes, soit {nb_min} minutes.")
		print("-" * 50)
		cv2.destroyAllWindows()
		for video in videos:
			video.release()

		# Ajout de l'audio de la vidéo de référence à la vidéo cible recolorisée
		audio_recupe = './videos/audio.wav'  # Chemin pour sauvegarder l'audio extrait
		for title in title_video:
			final_video_with_audio = title + ".mkv"  # Chemin de la vidéo finale avec l'audio ajouté
			video_with_audio = VideoSon(self.path_vid_tar, audio_recupe, final_video_with_audio, final_video_with_audio)
			video_with_audio.ajout_son_video()
