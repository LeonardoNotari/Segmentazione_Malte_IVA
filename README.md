**MMSFormer - Segmentazione delle malte**

Questo progetto estende il modello **MMSFormer** per affrontare il problema della **segmentazione delle malte**, con immagini acquisite in due modalità: **paralleli** e **incrociati**.
per poter addestrare il modello devono essere scaricati i al seguente linkhttps://drive.google.com/drive/folders/10XgSW8f7ghRs9fJ0dE-EV8G2E_guVsT5

---

## Modifiche principali rispetto al modello originale

1. **Dataset personalizzato**

   * È stato creato un nuovo file `dataset.py` nella cartella `semseg/dataset`.
   * Permette il caricamento del dataset delle malte con le modalità: paralleli e incrociati.

2. **Gestione della distribuzione non omogenea delle classi**

   * Possibilità di usare **Weighted Random Sampler** durante il training.

     * I pesi vengono calcolati **automaticamente** in base alla frequenza delle classi.
     * Attivabile nel file di configurazione con:

       ```
       WEIGHTED_RANDOM_SAMPLER: true
       ```
   * In alternativa, è possibile assegnare pesi personalizzati alle classi nel calcolo della loss.

     * Attivabile con:

       ```
       CLS_WEIGHTS: true
       CLASS_WEIGHTS: [w1, w2, w3]  
       ```

       I pesi delle classi sono da impostare manualmente tenendo conto che le classi rappresentano rispettivamente:

       * Legante: 70.46%
       * Porosità: 3.62%
       * Aggregati: 25.92%

3. **Funzionalità di testing**

   * Generazione di un file con **mappe delle predizioni** sul dataset di test.
   * Salvataggio delle immagini originali, label associate alle predizioni in un’unica cartella.
   * Possibilità di generare la **confusion matrix**.
   * Attivabili tramite configurazione:

     ```
     SAVE_PREDICTIONS: true
     SAVE_CONFUSION: true
     ```
   * Cartelle di salvataggio:

     ```
     VIS_SAVE_DIR: 'data/mortars_results'
     CONFUSION_DIR: 'data/confusion_matrix'
     ```

---

## Comandi principali

### Training

Per allenare il modello con la configurazione `mortars.yaml`:

```
python -m tools.train_mm --cfg configs/mortars.yaml
```

* Il modello verrà salvato in:

```
output/MMSFormer/MMSF-MORTARS-CONFIG
```

* Nome del modello:

```
MMSFormer_MMSFormer-B3_MORTARS_epochXX_XX.XX.pth
```

### Testing

Per testare il modello su un dataset di test:

```
python -m tools.val_mm --cfg configs/mortars.yaml
```

* Assicurarsi di aver aggiornato nel file `mortars.yaml` il percorso del modello salvato:

```
MODEL_PATH: 'output/MMSFormer/MMSF-MORTARS-CONFIG/MMSFormer_MMSFormer-B3_MORTARS_epochXX_XX.XX.pth'
```

---

## Esempio di configurazione (`mortars.yaml`)

```yaml
DEVICE          : cuda
SAVE_DIR        : 'output/MMSFormer'
GPUs            : 1
GPU_IDs         : [0]
USE_WANDB       : false
WANDB_NAME      : 'MMSF-MORTARS-CONFIG'

MODEL:
  NAME          : MMSFormer
  BACKBONE      : MMSFormer-B3
  PRETRAINED    : 'checkpoints/pretrained/segformer/mit_b3.pth'
  RESUME        : ''

DATASET:
  NAME          : MORTARS
  ROOT          : '/mortars_dataset'
  IGNORE_LABEL  : 3
  MODALS        : ['paralleli', 'incrociati'] 

TRAIN:
  IMAGE_SIZE    : [512, 512]      
  BATCH_SIZE    : 4
  EPOCHS        : 1
  EVAL_START    : 0
  EVAL_INTERVAL : 1
  AMP           : true
  DDP           : false
  WEIGHTED_RANDOM_SAMPLER : true

LOSS:
  NAME          : OhemCrossEntropy
  CLS_WEIGHTS   : false
  CLASS_WEIGHTS : [1.0, 8.0, 2.5] 

OPTIMIZER:
  NAME          : adamw
  LR            : 0.0001
  WEIGHT_DECAY  : 0.01

SCHEDULER:
  NAME          : warmuppolylr
  POWER         : 0.9
  WARMUP        : 10
  WARMUP_RATIO  : 0.1

EVAL:
  MODEL_PATH    : 'output/MMSFormer/MMSF-MORTARS-CONFIG/MMSFormer_MMSFormer-B3_MORTARS_epoch76_65.24.pth'
  IMAGE_SIZE    : [512, 512]
  BATCH_SIZE    : 1
  SAVE_PREDICTIONS : true
  SAVE_CONFUSION: true
  VIS_SAVE_DIR  : 'data/mortars_results'
  CONFUSION_DIR : 'data/confusion_matrix'
  MSF:  
    ENABLE      : false
    FLIP        : true
    SCALES      : [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
```


