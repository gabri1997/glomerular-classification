#  Classificazione delle caratteristiche glomerulari da WSI

##  Scopo del progetto

Lo scopo del progetto è quello di **classificare caratteristiche glomerulari** legate a tre aspetti principali:

- **Location**
- **Appearance**
- **Distribution**

Il processo parte dalle **Whole Slide Images (WSI)** originali fornite, da cui sono stati **automaticamente estratti i glomeruli** attraverso un modello **YOLO**.

##  Pipeline del progetto

### 1.  Estrazione dei glomeruli dalle WSI

- I glomeruli sono stati localizzati usando YOLO. Il processo di localizzazione è descritto nel documento `Report_Progetto_IgAN.docx`.
- Le **bounding box** rilevate sono state salvate in `Train_scripts/Annotations/`.
- A partire da queste coordinate, è stato eseguito lo script `Train_scripts/glomeruli_generator_from_wsi.py` per generare le patch dei glomeruli dalle WSI.
- I glomeruli estratti sono **1.421 immagini** in formato `.png`.  
 

### 2.  Assegnazione delle label ai glomeruli

Dopo l’estrazione, è stato necessario **trasferire le annotazioni cliniche** (riferite all’intera WSI) ai singoli glomeruli.

- Le annotazioni originali si trovano nel file: `Train_scripts/Excels/IF score.xlsx`

- La propagazione delle etichette è stata effettuata tramite lo script: `label_generator_from_excel.py`

> **Nota:** Le etichette sono state fornite per l'intera WSI, **non** per il singolo glomerulo.  
> Pertanto, a livello clinico, **non tutti i glomeruli** ereditano perfettamente la caratteristica osservata a livello globale.


## Strategia di aggregazione per la valutazione delle predizioni

Poiché le caratteristiche sono state annotate a livello WSI, è stato necessario definire un criterio per **riaggregare le predizioni** fatte a livello di singolo glomerulo. I medici hanno fornito le seguenti regole (solo per la classe Globale e Segmentale):

###  Classi "Segmentale" e "Globale"

- Una **WSI è considerata Globale** se **≥ 70%** dei glomeruli positivi (con intensità ≥1) presenta tale caratteristica.
- Una **WSI è considerata Segmentale** se **> 30%** dei glomeruli positivi presenta lesione segmentale  
  (quindi Globale sarà <= 70%).

###  Tutte le altre classi

Per le altre classi, una **WSI è considerata positiva** se:

- **≥ 50%** dei glomeruli appartenenti a quella WSI ha la **caratteristica predetta**, ma questa regola NON è stata fornita dai medici.

##  Obiettivo del modello

Il modello scelto per la classificazione è una **ResNet-18**, che verrà allenata sui **1.421 glomeruli estratti** per riconoscere le caratteristiche associate.  
Le predizioni saranno valutate sia a livello di glomerulo, sia con la logica di **aggregazione a livello WSI**.

## Download dei dati esterni

I dati di grandi dimensioni utilizzati in questo progetto non sono inclusi direttamente nel repository per motivi di spazio.

Puoi scaricare i dati da Google Drive (e i pesi del training per le varie classi) al seguente link:

[Download dataset (Google Drive)](https://drive.google.com/drive/folders/16ot_9aC8AH_lzr0gr3YE9ain1ZNVYlxk?usp=drive_link)  
[Download weights (Google Drive)](https://drive.google.com/drive/folders/1AqZdaK2FUIaQOHouKmhDmR4v9ubl0ZRX?usp=drive_link)

> **Nota:** Assicurati di avere i permessi necessari per accedere al file.  
> Se il link non funziona, contattami.

#  Replicazione Risultati - Segmentazione Glomeruli

##  Obiettivo del progetto

Questo repository contiene il codice e i dati utilizzati per **replicare i risultati** presentati nel seguente articolo, il codice è stato poi riadattato sui nuovi dati forniti dai medici:

 **Evaluation of the Classification Accuracy of the Kidney Biopsy Direct Immunofluorescence through Convolutional Neural Networks**  
🔗 **Link al paper**: [(https://pmc.ncbi.nlm.nih.gov/articles/PMC7536749/)]

---

##  Step 1 - Replicazione dei Risultati

L'obiettivo di questo primo step è stato **replicare i risultati pubblicati**, seguendo la stessa pipeline descritta nel paper.

###  Classe replicata: *MESANGIALE*

Per la classe **MESANGIALE**, sono stati utilizzati gli stessi file `.csv` presenti nel paper. Questi file sono stati salvati nella cartella:

### Contenuto della cartella `Files_old_Pollo`

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
I pesi si trovano nella cartella `Train_scripts/Paper_replicated_weights`

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

I risultati di tutti questi esperimenti si trovano nel file: 

`Train_scripts/Results/result_Seed16_MESANGIALE.json`

La cartella `Results` per il resto può essere ignorata.

###  K-Fold Cross-Validation

Per validare il modello in modo più robusto, è stata eseguita una **K-Fold Cross-Validation** con `k=4`, sempre sui nuovi dati. Sono stati generati 4 fold (split casuali) nella directory:

`Train_scripts/Base_split_over_wsi/Cross_fold/`

Per motivi di spazio, i **pesi generati** dal training sui 4 fold **non sono stati salvati** nel repository.

I risultati dei test sui 4 fold per ciascuna classe sono stati salvati in:

`Train_scripts/Results_folds_test`

Per aggregare i risultati provenienti dai vari fold, uso lo script Python:

`Train_scripts/Results/aggregate_fold_results.py`

> Questo script consente di raccogliere i risultati dei 4 esperimenti e ottenere una media delle metriche più rilevanti con la loro deviazione standard.

### Altri file

- `Train_scripts/Results/result_MESANGIALE.json`: contiene prove preliminari effettuate durante la fase di replicazione del paper.  
  > **Può essere ignorato**.

## Risultati finali a livello dei glomeruli - Cross Validation (4 Fold)

I risultati finali ottenuti tramite k-fold cross-validation con `k=4` sono riassunti nella seguente tabella:

| Classe                           | Accuracy       | Precision      | Recall         | F1-Score       |
|----------------------------------|----------------|----------------|----------------|----------------|
| MESANGIALE                       | 0.717 ± 0.050  | 0.822 ± 0.134  | 0.742 ± 0.090  | 0.768 ± 0.034  |
| GRAN_GROSS                       | 0.662 ± 0.084  | 0.706 ± 0.120  | 0.707 ± 0.040  | 0.703 ± 0.071  |
| GRAN_FINE                        | 0.704 ± 0.041  | 0.593 ± 0.104  | 0.576 ± 0.164  | 0.574 ± 0.111  |
| PARETE REGOLARE DISCONTINUA      | —              | —              | —              | —              |
| PARETE REGOLARE CONTINUA         | 0.734 ± 0.056  | 0.620 ± 0.202  | 0.645 ± 0.151  | 0.624 ± 0.162  |
| PARETE IRREGOLARE                | 0.646 ± 0.057  | 0.585 ± 0.022  | 0.600 ± 0.088  | 0.590 ± 0.048  |
| GLOBALE                          | 0.798 ± 0.107  | 0.880 ± 0.068  | 0.884 ± 0.114  | 0.879 ± 0.072  |
| SEGMENTALE                       | 0.773 ± 0.140  | 0.202 ± 0.211  | 0.132 ± 0.172  | 0.107 ± 0.084  |

> *I valori rappresentano media e deviazione standard delle metriche calcolate su 4 diverse suddivisioni del dataset.*
Per la classe PARETE REGOLARE DISCONTINUA (che nel file IF score.xlsx è capillary wall regular discontinuous) non ci sono abbastanza esempi positivi. 

**Nota importante**: le etichette per le classi **GLOBALE** e **SEGMENTALE** sono state ottenute per aggregazione:

- **SEGMENTALE** = `True` se almeno una delle due etichette è `True`:
  - *diffuse/segmental*  
  - *focal/segmental*  

- **GLOBALE** = `True` se almeno una delle due etichette è `True`:
  - *diffuse/global*  
  - *focal/global*  

Mappatura delle labels tra i nomi delle classi finali e quelli presenti nel file Excel (mappatura presente anche nello script `label_generator_from_excel.py`):

| Nome Excel         | Nome classe finale                                      |
|--------------------|---------------------------------------------------------|
| `LIN`              | linear                                                  |
| `PSEUDOLIN`        | pseudolinear                                            |
| `GRAN_GROSS`       | coarse granular                                         |
| `GRAN_FINE`        | fine granular                                           |
| `GEN_SEGM`         | diffuse/segmental                                       |
| `GEN_DIFF`         | diffuse/global                                          |
| `FOC_SEGM`         | focal/segmental                                         |
| `FOC_GLOB`         | focal/global                                            |
| `MESANGIALE`       | mesangial                                               |
| `PAR_REGOL_CONT`   | continuous regular capillary wall (subendothelial)      |
| `PAR_REGOL_DISCONT`| capillary wall regular discontinuous                    |
| `PAR_IRREG`        | irregular capillary wall (subendothelial)               |
| `INTENS`           | INTENSITY                                               |

 
## Aggregazione dei risultati

Usando lo script `from_prediction_to_final_excel.py` (presente nella cartella `Results_folds_test`), i risultati ottenuti sui glomeruli vengono **aggregati seguendo le regole proposte da Magistroni**.    
Per ciascuna classe, lo script genera nelle sottocartelle di `Results_folds_test` i file `allfolds_aggregated.csv`.

Successivamente, lo script `results_comparison.py` (anch’esso in `Results_folds_test`) confronta i risultati ottenuti con il file Excel fornito dai medici.  
Per semplicità, si parte da una versione del file già pre-manipolata (le annotazioni dei medici originali sono nel file `Train_scripts/Excels/If-score.xlsx`):  

`Prettified_scores_total_wsi_classification_Lv0_magistroni_norm_IF.xlsx`

Lo script calcola inoltre le **metriche aggregate** e produce come output il file Excel finale: `WSI_Score.xlsx`

## Risultati finali

| Class      | Accuracy | Precision | Recall | F1    |
|------------|----------|-----------|--------|------ |
| Coarse     | 0.697    | 0.795     | 0.738  | 0.765 |
| Fine       | 0.745    | 0.565     | 0.481  | 0.520 |
| Segmental  | 0.761    | 0.250     | 0.333  | 0.286 |
| Global     | 0.729    | 0.833     | 0.828  | 0.831 |
| Irregular  | 0.612    | 0.608     | 0.533  | 0.568 |
| Continuous | 0.798    | 0.617     | 0.592  | 0.604 |
| Mesangial  | 0.750    | 0.892     | 0.759  | 0.820 |






