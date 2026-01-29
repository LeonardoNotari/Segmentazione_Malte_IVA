import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_confusion_matrix(cm, class_names, normalize=True, save_path=None):
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


if __name__ == "__main__":
    cm_file = Path("data/confusion_matrix/confusion_matrix.npy")
    cm = np.load(cm_file)
    class_names = ["Legante", "Porosità", "Aggregati"] 
    plot_confusion_matrix(cm, class_names, normalize=True, save_path="data/confusion_matrix/confusion_matrix.png")


