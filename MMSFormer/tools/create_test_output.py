from PIL import Image
import numpy as np
import os
import shutil


cartella_predizioni = "MMSFormer/data/mortars_results"
cartella_label = "mortars_dataset/label"
cartella_immagini_i = "mortars_dataset/incrociati"
cartella_immagini_p = "mortars_dataset/paralleli"

cartella_output = "predictions_check"
os.makedirs(cartella_output, exist_ok=True)

estensioni = (".png", ".tif")

predizioni = {f[:-4] for f in os.listdir(cartella_predizioni) if f.lower().endswith(estensioni)}
labels = {f[:-4] for f in os.listdir(cartella_label) if f.lower().endswith(estensioni)}
immagini_i = {f[:-4] for f in os.listdir(cartella_immagini_i) if f.lower().endswith(estensioni)}
immagini_p = {f[:-4] for f in os.listdir(cartella_immagini_p) if f.lower().endswith(estensioni)}
print(predizioni)
file_comuni = predizioni & labels & immagini_i & immagini_p
print(f"Trovati {len(file_comuni)} set completi.")

# Mappa colori
color_map = {
    0: (0, 0, 0),   #legante nero
    1: (255, 0, 0), #porosità rosso
    2: (0, 255, 0), #aggregati verdi
    3: (0, 0, 255), #ignore blu
}

for nome in file_comuni:
    # Predizioni
    shutil.copy(os.path.join(cartella_predizioni, nome+".png"),
                os.path.join(cartella_output, f"{nome}_predizione.png"))
    
    #label
    label_path = os.path.join(cartella_label, nome+".tif")
    label_img = Image.open(label_path).convert("L")  # <-- forza 2D
    label_array = np.array(label_img)

    color_label = np.zeros((label_array.shape[0], label_array.shape[1], 3), dtype=np.uint8)
    
    for val, color in color_map.items():
        mask = label_array == val
        color_label[mask, 0] = color[0]
        color_label[mask, 1] = color[1]
        color_label[mask, 2] = color[2]

    color_label_img = Image.fromarray(color_label)
    color_label_img.save(os.path.join(cartella_output, f"{nome}_label.tif"))
    
    # immagini nelle due modalità
    shutil.copy(os.path.join(cartella_immagini_i, nome+".tif"),
                os.path.join(cartella_output, f"{nome}_incrociati.tif"))
    shutil.copy(os.path.join(cartella_immagini_p, nome+".tif"),
                os.path.join(cartella_output, f"{nome}_paralleli.tif"))

print("Copia, rinomina e colorazione completata!")

