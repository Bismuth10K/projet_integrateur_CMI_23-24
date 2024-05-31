from moviepy.editor import *


class VideoSon:
	# fonction d'initialisation
	def __init__(self, video1, audio_recupe, video2, final_video):
		"""
		Parameters
		----------
		video1 : chemin de la video de base
		audio_recupe : chemin de l'audio de la video de base
		video2 : chemin de la video cible
		final_video : chemin de video cible avec le nouveau son
		"""
		self.video1 = video1  # chemin de la video de base
		self.audio_recupe = audio_recupe  # chemin de l'audio de la video de base
		self.video2 = video2  # chemin de la video cible
		self.final_video = final_video  # chemin de video cible avec le nouveau son

	# fonction qui permet de récupérer le son de la video
	def recuper_son_video(self):
		"""
		Extrait le son d'une vidéo et le sauvegarde dans un fichier audio.

		Returns
		-------
		l'audio de la vidéo
		"""
		# Chargement de la vidéo
		video = VideoFileClip(self.video1)
		# Extraction de l'audio
		audio = video.audio
		# Sauvegarde de l'audio dans un fichier
		audio.write_audiofile(self.audio_recupe, codec="pcm_s16le")

		# Fermeture de la vidéo pour libérer la mémoire
		video.close()

		return self.audio_recupe

	# fonction qui permet de mettre le son sur une autre video
	def ajout_son_video(self):
		"""
		Ajoute une piste audio à une vidéo et sauvegarde le résultat dans un fichier.

		Returns
		-------
		la vidéo à laquelle on a ajouté l'audio

		"""

		# Récupérer l'audio de la première vidéo
		chemin_audio = self.recuper_son_video()

		# Charger la vidéo sans son
		video = VideoFileClip(self.video2)

		# Charger l'audio extrait
		audio = AudioFileClip(chemin_audio)

		# Associer le nouvel audio à la vidéo
		video = video.set_audio(audio)

		# Écrire la nouvelle vidéo avec l'audio ajouté
		video.write_videofile(self.final_video, codec="libx264", audio_codec="aac")

		# Fermeture de l'audio pour libérer la mémoire
		audio.close()

		return self.final_video
