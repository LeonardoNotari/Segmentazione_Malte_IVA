import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_confusion_matrix(cm, class_names, normalize=True, save_path=None):
    """
    cm: confusion matrix numpy array (n_classes x n_classes)
    class_names: lista di nomi delle classi
    normalize: se True normalizza i valori tra 0 e 1
    save_path: percorso opzionale per salvare l'immagine
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm)  # evita divisioni per zero

    plt.figure(figsize=(8, 8))
    plt.imshow(cm, cmap='Blues')
    plt.colorbar()
    plt.xticks(range(len(class_names)), class_names, rotation=45)
    plt.yticks(range(len(class_names)), class_names)

    # Scrivi i valori all'interno delle celle
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            plt.text(j, i, f"{cm[i, j]:.2f}", ha='center', va='center', color='red')

    plt.title("Confusion Matrix")
    #plt.xlabel("Predicted")
    #plt.ylabel("Ground Truth")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()


# -------------------- Main --------------------

if __name__ == "__main__":
    # Path della confusion matrix salvata
    cm_file = Path("data/confusion_matrix/confusion_matrix.npy")
    cm = np.load(cm_file)

    # Lista dei nomi delle classi (modifica secondo il tuo dataset)
    class_names = ["Legante", "Porosità", "Aggregati"]  # esempio, metti le tue

    # Visualizza e salva l'immagine
    plot_confusion_matrix(cm, class_names, normalize=True, save_path="data/confusion_matrix/confusion_matrix.png")
