import time
from tqdm import tqdm
import matplotlib

from POTI import POTI
from POTV import POTV
from automatisation_series import automate_series


def runner(path_imgs_ref: list[str], path_vid_tar: str, list_method: list[str] = ("emd", "sinkhorn", "linear", "gaussian")):
	"""
	La fonction qui automatise le traitement de plusieurs vidéos d'un coup.

	Parameters
	----------
	path_imgs_ref (list) : Liste des chemins des images références
	path_vid_tar (str) : Chemin de la vidéo cible
	list_method (list) : Liste des méthodes appliquées
	"""
	print("Lancement en cours, ne pas s'inquiéter.")
	start = time.time()
	list_model = []
	potv1 = POTV(path_imgs_ref[0], path_vid_tar)
	list_model.extend([potv1.train_ot(method) for method in list_method])

	for img in range(1, len(path_imgs_ref)):
		potv1.set_reference(path_imgs_ref[img])
		list_model.extend([potv1.train_ot(method) for method in list_method])
	nb_sec = round(time.time() - start, 2)
	nb_min = round(nb_sec / 60, 2)
	print(f"Temps d'entraînement total : {nb_sec}s, soit {nb_min}min.")

	titles = ["render_" + img.split("/")[-1][:-4] + "_" + method for img in path_imgs_ref for method in list_method]
	potv1.apply_ot(list_model, titles)
	nb_sec = round(time.time() - start, 2)
	nb_min = round(nb_sec / 60, 2)
	print(f"Temps totale de render : {nb_sec}s, soit {nb_min}min.")


if __name__ == '__main__':
	print("Quelle version voulez-vous exécuter ?")
	print("0 - Train à la ciotat (POTV)")
	print("1 - Video city (POTV)")
	print("2 - Control vers picture_city (POTI)")
	print("3 - Cathedrale vers Control (POTI)")
	print("4 - Séries des Cathédrales de Rouen (4 méthodes) (automatisation_series)")

	match int(input("Votre choix : ")):
		case 0:
			img_ref = ['./ciotat/fresque.jpg', './ciotat/marseille.jpg', './ciotat/ref_ciotat_colorise.jpg',
					   './ciotat/VoieFerree.jpg']
			vid_tar = './ciotat/Arrivee train à La Ciotat.mp4'
			runner(img_ref, vid_tar)
		case 1:
			img_ref = ['./photos/picture_city.jpg',
					   './photos/cathedrale_rouen_monet/Cathédrale de Rouen, façade ouest.jpg']
			vid_tar = './videos/ultra_short_city.mp4'
			runner(img_ref, vid_tar, ["sinkhorn", "emd"])
		case 2:
			img_ref = './photos/control_game_red_room.jpg'
			img_tar = './photos/picture_city.jpg'

			poti1 = POTI(img_ref, img_tar)
			poti1.plot_photos()
			poti1.plot_distributions()
			poti1.train_ot()
			matplotlib.image.imsave('control_onto_picture_city.png', poti1.apply_ot())
		case 3:
			img_ref = './photos/cathedrale_rouen_monet/Cathédrale de Rouen, façade ouest.jpg'
			img_tar = './photos/control_game_red_room.jpg'

			poti1 = POTI(img_ref, img_tar)
			poti1.plot_photos()
			poti1.plot_distributions()
			poti1.train_ot()
			matplotlib.image.imsave('cathedrale_onto_control.png', poti1.apply_ot())
		case 4:
			automate_series("./photos/cathedrale_rouen_monet", "serie_cathedrale_monet", method="emd")
			automate_series("./photos/cathedrale_rouen_monet", "serie_cathedrale_monet", method="sinkhorn")
			automate_series("./photos/cathedrale_rouen_monet", "serie_cathedrale_monet", method="linear")
			automate_series("./photos/cathedrale_rouen_monet", "serie_cathedrale_monet", method="gaussian")
		case _:
			print("Il n'y a rien associé à ce numéro.")
