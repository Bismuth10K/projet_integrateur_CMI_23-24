import os
import time

import matplotlib.image
import matplotlib.pyplot as plt

from POTO import POTO
from fonctions import import_image


def automate_series(path_series: str, name_render: str):
	start = time.time()
	list_obj_poto = []
	list_info_imgs = []
	list_name = os.listdir(path_series)
	for file in list_name:
		if file.endswith(".jpg") or file.endswith(".jpeg"):
			print(f"---Clustering de '{file.title()[:-4]}' (cluster {len(list_obj_poto) + 1})---")
			img, mat = import_image(os.path.join(path_series, file))
			cur_poto = POTO(img)
			list_info_imgs.append(cur_poto.get_ref_all())
			list_obj_poto.append(cur_poto)

	path_render = os.path.join(path_series + "/individual_render/")
	if not os.path.exists(path_render):
		os.makedirs(path_render)
	fig, axs = plt.subplots(len(list_obj_poto), len(list_obj_poto), figsize=(10, 8))
	i = 0
	for poto in list_obj_poto:
		j = 0
		for info in list_info_imgs:
			print(f"- Image référence {i + 1} ({list_name[i]}) sur image cible {j + 1} ({list_name[j]})")
			if i != j:
				poto.set_target_all(info[0], info[1], info[2], info[3])
				poto.train_ot()
				img_recolored = poto.apply_ot()
				axs[i, j].imshow(img_recolored)

				matplotlib.image.imsave(os.path.join(path_render, f"from{i + 1}_to{j + 1}.png"), img_recolored)
			else:
				axs[i, j].imshow(info[0])
			axs[i, j].set(xlabel=f'{j + 1}', ylabel=f'{i + 1}')
			axs[i, j].label_outer()
			axs[i, j].spines[['top', 'right', 'left', 'bottom']].set_visible(False)
			axs[i, j].tick_params(left=False, right=False, labelleft=False,
								labelbottom=False, bottom=False)
			j += 1
		i += 1
	fig.suptitle("Série 'Cathédrale de Rouen' par Monet")
	plt.savefig(os.path.join(path_render, f"{name_render}.png"), bbox_inches='tight')

	nb_sec = round(time.time() - start, 2)
	nb_min = round(nb_sec / 60, 2)
	print(f"Durée totale : {nb_sec}s - {nb_min}min")

	titre = ""
	for i in range(len(list_name)):
		titre += f"{i + 1} : {list_name[i]}\n"
	with open(os.path.join(path_render, f"legende_{name_render}.txt"), 'w') as f:
		f.write(titre + "\n")
		f.write("Une ligne contient toutes les images.\n")
		f.write("Les couleurs d'une image i sont appliquées à l'ensemble des images (sauf elle même, image à la "
				"position (i, i))sur la ligne i.\n")
		f.write(f"Temps d'exécution et de render = {nb_sec} secondes, soit {nb_min} minutes.")


if __name__ == '__main__':
	automate_series("./photos/cathedrale_rouen_monet", "serie_cathedrale_monet")
