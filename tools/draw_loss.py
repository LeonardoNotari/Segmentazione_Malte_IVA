import json
import matplotlib.pyplot as plt

with open("output/MMSFormer/MMSF-MORTARS-CONFIG-PADDED/train_loss.json") as f:
    losses = json.load(f)

plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("MMSFormer Training Loss")
plt.grid(True)
plt.savefig("output/MMSFormer/MMSF-MORTARS-CONFIG_PADDED/train_loss.png")

    
    
    