#  Classificazione delle caratteristiche glomerulari da WSI

##  Scopo del progetto

Lo scopo del progetto è quello di **classificare caratteristiche glomerulari** legate a tre aspetti principali:

- **Location**
- **Appearance**
- **Distribution**

Il processo parte dalle **Whole Slide Images (WSI)** originali fornite, da cui sono stati **automaticamente estratti i glomeruli** attraverso un modello **YOLO**.

##  Pipeline del progetto

### 1.  Estrazione dei glomeruli dalle WSI

- I glomeruli sono stati localizzati usando YOLO.
- Le **bounding box** rilevate sono state salvate in: `Train_scripts/Annotations/`

- A partire da queste coordinate, tramite lo script: `Train_scripts/glomeruli_generator_from_wsi.py`

sono state generate le immagini dei glomeruli a **livello 0 della WSI**.

- I glomeruli estratti sono **1.421 immagini** in formato `.png`.  
 

### 2.  Assegnazione delle label ai glomeruli

Dopo l’estrazione, è stato necessario **trasferire le annotazioni cliniche** (riferite all’intera WSI) ai singoli glomeruli.

- Le annotazioni originali si trovano nel file: `Train_scripts/Excels/IF score.xlsx`

- La propagazione delle etichette è stata effettuata tramite lo script:


> ⚠️ Le etichette sono state fornite per l'intera WSI, **non** per il singolo glomerulo.  
> Pertanto, **non tutti i glomeruli** ereditano perfettamente la caratteristica osservata a livello globale.


## Strategia di aggregazione per la valutazione delle predizioni

Poiché le caratteristiche sono state annotate a livello WSI, è stato necessario definire un criterio per **riaggregare le predizioni** fatte a livello di singolo glomerulo. I medici hanno fornito le seguenti regole:

###  Classi "Segmentale" e "Globale"

- Una **WSI è considerata Globale** se **≥ 70%** dei glomeruli positivi (con intensità ≥1) presenta tale caratteristica.
- Una **WSI è considerata Segmentale** se **> 30%** dei glomeruli positivi presenta lesione segmentale  
  (quindi Globale sarà < 70%).

###  Tutte le altre classi

Per le altre classi, una **WSI è considerata positiva** se:

- **≥ 50%** dei glomeruli appartenenti a quella WSI ha la **caratteristica predetta**.

##  Obiettivo del modello

Il modello scelto per la classificazione è una **ResNet-18**, che verrà allenata sui **1.421 glomeruli estratti** per riconoscere le caratteristiche associate.  
Le predizioni saranno valutate sia a livello di glomerulo, sia con la logica di **aggregazione a livello WSI**.


## Download dei dati esterni

I dati di grandi dimensioni utilizzati in questo progetto non sono inclusi direttamente nel repository per motivi di spazio.

Puoi scaricare i dati da Google Drive al seguente link:

[Download dataset (Google Drive)](https://drive.google.com/drive/folders/16ot_9aC8AH_lzr0gr3YE9ain1ZNVYlxk?usp=drive_link)

> ⚠️ **Nota:** Assicurati di avere i permessi necessari per accedere al file.  
> Se il link non funziona, contattami.

#  Replicazione Risultati - Segmentazione Glomeruli

##  Obiettivo del progetto

Questo repository contiene il codice e i dati utilizzati per **replicare i risultati** presentati nel seguente articolo:

 **Evaluation of the Classification Accuracy of the Kidney Biopsy Direct Immunofluorescence through Convolutional Neural Networks**  
🔗 **Link al paper**: [(https://pmc.ncbi.nlm.nih.gov/articles/PMC7536749/)]

---

##  Step 1 - Replicazione dei Risultati

L'obiettivo di questo primo step è stato **replicare i risultati pubblicati**, seguendo la stessa pipeline descritta nel paper.

###  Classe replicata: *MESANGIALE*

Per la classe **MESANGIALE**, sono stati utilizzati gli stessi file `.csv` presenti nel paper. Questi file sono stati salvati nella cartella:

### 📁 Contenuto della cartella `Files_old_Pollo`

| File                    | Descrizione                                  |
|-------------------------|----------------------------------------------|
| `train_mesangiale.csv`  | Dataset di training                          |
| `val_mesangiale.csv`    | Dataset di validazione                       |
| `test_mesangiale.csv`   | Dataset di test     

### Parametri di replicazione

I parametri rilevanti con cui sono stati replicati i risultati sui dati del paper sono i seguenti:

| Parametro        | Valore     |
|------------------|------------|
| `dropout`        | `false`    |
| `sampler`        | `false`    |
| `classes`        | `2`        |
| `wloss`          | `true`     |
| `batch_size`     | `64`       |
| `learning_rate`  | `0.1`      |
| `epochs`         | `180`      |
| `size`           | `512`      |
| `w4k`            | `true`     |
| `wdiapo`         | `false`    |
| `augm_config`    | `0`        |
| `seed`           | `42`       |


Questi pesi sono stati utilizzati per valutare le prestazioni del modello replicato sui dati del paper.

### Metriche ottenute

| Metrica     | Valore     |
|-------------|------------|
| Accuracy    | 0.817      |
| Precision   | 0.680      |
| Recall      | 0.737      |
| F1-score    | 0.707      |

##  Step 2 - Esperimenti su nuove WSI

Dopo aver replicato i risultati del paper originale (Step 1), sono stati condotti diversi esperimenti utilizzando la classe **MESANGIALE** su **glomeruli estratti da nuove Whole Slide Images (WSI)**.

###  Esperimenti condotti

Gli esperimenti includono le seguenti variazioni e tecniche:

- Uso del **Weighted Sampler** per bilanciare le classi
- Scelta e confronto tra diversi **optimizer** e **scheduler**
- **Tuning degli iperparametri**, in particolare:
  - `learning rate`
  - `batch size`
- Monitoraggio di **loss di validazione**, **valore del learning rate** e altre metriche tramite `wandb`
- Uso di **loss pesata** in base alla distribuzione delle classi nel dataset
- **Fine-tuning** del modello usando i pesi ottenuti dal training sui dati del paper
- **Retraining completo** (senza freeze dei layer) sulle nuove WSI

Tutti gli esperimenti sono stati **loggati e confrontati su Weights & Biases (wandb)**.

###  Scelta del miglior setting

Dopo aver confrontato gli esperimenti eseguiti su uno **split singolo random** (con `seed = 16`), è stato selezionato il miglior set di parametri sulla base dei grafici ottenuti.

I risultati di questo esperimento si trovano nel file: 

`Train_scripts/Results/result_Seed16_MESANGIALE.json`

###  K-Fold Cross-Validation

Per validare il modello in modo più robusto, è stata eseguita una **K-Fold Cross-Validation** con `k=4`, sempre sui nuovi dati. Sono stati generati 4 fold (split casuali) nella directory:

`Train_scripts/Base_split_over_wsi/Cross_fold/`

Per motivi di spazio, i **pesi generati** dal training sui 4 fold **non sono stati salvati** nel repository.

I risultati dei test sui 4 fold sono stati salvati in:

`Train_scripts/Results/result_FoldSeed42_[['MESANGIALE']].json`

Per aggregare i risultati provenienti dai vari fold, è disponibile uno script Python:

`Train_scripts/Results/aggregate_fold_results.py`

> Questo script consente di raccogliere i risultati dei 4 esperimenti e ottenere una media delle metriche più rilevanti.

### 📂 Altri file

- `Train_scripts/Results/result_MESANGIALE.json`: contiene prove preliminari effettuate durante la fase di replicazione del paper.  
  > ⚠️ **Può essere ignorato**.




