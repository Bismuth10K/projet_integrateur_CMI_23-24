import os
import time

import matplotlib.pyplot as plt

from POTO import POTO
from fonctions import import_image


def automate_series(path_series: str, name_render: str):
	start = time.time()
	list_obj_poto = []
	list_info_imgs = []
	list_name = os.listdir((path_series))
	for file in list_name:
		if file.endswith(".jpg") or file.endswith(".jpeg"):
			print(f"---Clustering de '{file.title()[:-4]}' (cluster {len(list_obj_poto) + 1})---")
			img, mat = import_image(os.path.join(path_series, file))
			cur_poto = POTO(img)
			list_info_imgs.append(cur_poto.get_ref_all())
			list_obj_poto.append(cur_poto)

	fig, axs = plt.subplots(len(list_obj_poto), len(list_obj_poto), figsize=(10, 8))
	i = 0
	for poto in list_obj_poto:
		j = 0
		for info in list_info_imgs:
			print(f"- Image référence {i + 1} ({list_name[i]}) sur image cible {j + 1} ({list_name[j]})")
			if i != j:
				poto.set_target_all(info[0], info[1], info[2])
				poto.train_ot()
				axs[i, j].imshow(poto.apply_ot())
			else:
				axs[i, j].imshow(info[0])
			axs[i, j].set(xlabel=f'{j+1}', ylabel=f'{i+1}')
			axs[i, j].label_outer()
			axs[i, j].spines[['top', 'right', 'left', 'bottom']].set_visible(False)
			axs[i, j].tick_params(left=False, right=False, labelleft=False,
							labelbottom=False, bottom=False)
			j += 1
		i += 1
	fig.suptitle("Série 'Cathédrale de Rouen' par Monet")

	titre = ""
	for i in range(len(list_name)):
		titre += f"{i + 1} : {list_name[i]}\n"
	with open(os.path.join(path_series, f"legende_{name_render}.txt"), 'w') as f:
		f.write(titre)
	plt.savefig(os.path.join(path_series, f"{name_render}.png"), bbox_inches='tight')
	# plt.show()
	nb_sec = round(time.time() - start, 2)
	nb_min = round(nb_sec / 60, 2)
	print(f"Durée totale : {nb_sec}s - {nb_min}min")


if __name__ == '__main__':
	automate_series("./photos/cathedrale_rouen_monet", "serie_cathedrale_monet")
