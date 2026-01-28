from PIL import Image
import numpy as np
import os
import shutil

# Percorsi
cartella_predizioni = "MMSFormer/data/mortars_results"
cartella_label = "mortars_dataset/label"
cartella_immagini = "mortars_dataset/incrociati"

# Output
cartella_output = "output"
os.makedirs(cartella_output, exist_ok=True)

# Estensioni
estensioni = (".png", ".tif")

predizioni = {f[:-4] for f in os.listdir(cartella_predizioni) if f.lower().endswith(estensioni)}
labels = {f[:-4] for f in os.listdir(cartella_label) if f.lower().endswith(estensioni)}
immagini = {f[:-4] for f in os.listdir(cartella_immagini) if f.lower().endswith(estensioni)}


file_comuni = predizioni & labels & immagini
print(f"Trovati {len(file_comuni)} set completi.")

# Mappa colori
color_map = {
    0: (0, 0, 0),
    1: (255, 0, 0),
    2: (0, 255, 0),
    3: (0, 0, 255),
}

for nome in file_comuni:
    # Predizione
    shutil.copy(os.path.join(cartella_predizioni, nome+".png"),
                os.path.join(cartella_output, f"{nome}_predizione.png"))
    
    # Label: apri in scala di grigi
    label_path = os.path.join(cartella_label, nome+".tif")
    label_img = Image.open(label_path).convert("L")  # <-- forza 2D
    label_array = np.array(label_img)
    
    # Array RGB vuoto
    color_label = np.zeros((label_array.shape[0], label_array.shape[1], 3), dtype=np.uint8)
    
    # Applica la color map
    for val, color in color_map.items():
        mask = label_array == val
        color_label[mask, 0] = color[0]
        color_label[mask, 1] = color[1]
        color_label[mask, 2] = color[2]
    
    # Salva
    color_label_img = Image.fromarray(color_label)
    color_label_img.save(os.path.join(cartella_output, f"{nome}_label.tif"))
    
    # Immagine originale
    shutil.copy(os.path.join(cartella_immagini, nome+".tif"),
                os.path.join(cartella_output, f"{nome}_immagine.tif"))

print("Copia, rinomina e colorazione completata!")
