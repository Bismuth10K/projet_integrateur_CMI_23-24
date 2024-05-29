"""
objet : affichage d'une barre de progression
rôle : indique le numéro de la frame sur laquelle le clustering se fait et à quelle pourcentage cela correspond t-il
"""
from tqdm import tqdm
import time
for i in tqdm (range(10), desc="Work in progress", unit="%", unit_scale=True, leave=True):
    time.sleep(1)

