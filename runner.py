from POTV import POTV
import time

if __name__ == '__main__':
	start = time.time()
	img_ref = ['./ciotat/fresque.jpg',
			   './ciotat/marseille.jpg',
			   './ciotat/ref_ciotat_colorise.jpg',
			   './ciotat/VoieFerree.jpg']
	vid_tar = './ciotat/Arrivee train à La Ciotat.mp4'

	list_method = ["emd", "sinkhorn", "linear", "gaussian"]

	list_model = []
	potv1 = POTV(img_ref[0], vid_tar)
	list_model.extend([potv1.train_ot(method) for method in list_method])

	for img in range(1, len(img_ref)):
		potv1.set_reference(img_ref[img])
		list_model.extend([potv1.train_ot(method) for method in list_method])
	nb_sec = round(time.time() - start, 2)
	nb_min = round(nb_sec / 60, 2)
	print(f"Temps d'entraînement total : {nb_sec}s, soit {nb_min}min.")

	titles = ["./ciotat/render/" + img.split("/")[-1][:-4] + "_" + method for img in img_ref for method in list_method]
	potv1.apply_ot(list_model, titles)
	nb_sec = round(time.time() - start, 2)
	nb_min = round(nb_sec / 60, 2)
	print(f"Temps totale de render : {nb_sec}s, soit {nb_min}min.")
