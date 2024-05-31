import os
import time

import matplotlib.image
import matplotlib.pyplot as plt

from POTI import POTI


def automate_series(path_series: str, name_render: str, method: str = "emd"):
	"""
	Cette fonction automatise le render d'une série d'images.
	Elle crée toutes les images recolorisées et les enregistre séparément et en groupe sur un plot spécifique.
	Un paramètre pourra être fait pour gérer les rendus (si on veut des individuelles ou pas).

	En plus de ces images, elle génère un txt avec la légende des images et des informations complementaires tells que
	le temps de compilation.

	Parameters
	----------
	path_series (str) : chemin vers les images
	name_render (str) : nom du plot avec toutes les images
	"""
	start = time.time()
	list_obj_poti = []
	list_info_imgs = []
	list_name = os.listdir(path_series)

	titre = ""
	file = 0
	while file < len(list_name):
		if list_name[file].endswith(".jpg") or list_name[file].endswith(".jpeg"):
			print(f"---Clustering de '{list_name[file].title()[:-4]}' (cluster {len(list_obj_poti) + 1})---")
			cur_poti = POTI(os.path.join(path_series, list_name[file]))
			list_info_imgs.append(cur_poti.get_ref_all())
			list_obj_poti.append(cur_poti)
			titre += f"{file + 1} : {list_name[file]}\n"
			file += 1
		else:
			list_name.remove(list_name[file])

	path_render = os.path.join(path_series + f"/individual_render_{method}/")
	if not os.path.exists(path_render):
		os.makedirs(path_render)
	plt.rcParams['figure.dpi'] = 300
	plt.rcParams['savefig.dpi'] = 300
	fig, axs = plt.subplots(len(list_obj_poti), len(list_obj_poti), figsize=(10, 8))
	i = 0
	for poti in list_obj_poti:
		j = 0
		for info in list_info_imgs:
			print(f"- Image référence {i + 1} ({list_name[i]}) sur image cible {j + 1} ({list_name[j]})")
			if i != j:
				poti.set_target_all(info[0], info[1], info[2], info[3])
				poti.train_ot(method)
				img_recolored = poti.apply_ot()
				axs[i, j].imshow(img_recolored)

				matplotlib.image.imsave(os.path.join(path_render, f"from{i + 1}_to{j + 1}_method-{method}.png"),
										img_recolored)
			else:
				axs[i, j].imshow(info[0])
			axs[i, j].set(xlabel=f'{j + 1}', ylabel=f'{i + 1}')
			axs[i, j].label_outer()
			axs[i, j].spines[['top', 'right', 'left', 'bottom']].set_visible(False)
			axs[i, j].tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
			j += 1
		i += 1
	fig.suptitle("Série 'Cathédrale de Rouen' par Monet - " + method)
	plt.savefig(os.path.join(path_render, f"{name_render}.png"), bbox_inches='tight')

	nb_sec = round(time.time() - start, 2)
	nb_min = round(nb_sec / 60, 2)
	print(f"Durée totale : {nb_sec}s - {nb_min}min")

	with open(os.path.join(path_render, f"legende_{name_render}_{method}.txt"), 'w') as f:
		f.write(titre + "\n")
		f.write("Une ligne contient toutes les images.\n")
		f.write("Les couleurs d'une image i sont appliquées à l'ensemble des images (sauf elle même, l'image à la "
				"position (i, i)) sur la ligne i.\n")
		f.write(f"Temps d'exécution et de render = {nb_sec} secondes, soit {nb_min} minutes.")
