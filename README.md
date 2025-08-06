## Download dei dati esterni

I dati di grandi dimensioni utilizzati in questo progetto non sono inclusi direttamente nel repository per motivi di spazio.

Puoi scaricare i dati da Google Drive al seguente link:

[Download dataset (Google Drive)](https://drive.google.com/drive/folders/16ot_9aC8AH_lzr0gr3YE9ain1ZNVYlxk?usp=drive_link)

> ⚠️ **Nota:** Assicurati di avere i permessi necessari per accedere al file.  
> Se il link non funziona, contattami.

#  Replicazione Risultati - Segmentazione Glomeruli

##  Obiettivo del progetto

Questo repository contiene il codice e i dati utilizzati per **replicare i risultati** presentati nel seguente articolo scientifico:

 **Evaluation of the Classification Accuracy of the Kidney Biopsy Direct Immunofluorescence through Convolutional Neural Networks**  
🔗 **Link al paper**: [(https://pmc.ncbi.nlm.nih.gov/articles/PMC7536749/)]

---

##  Step 1 - Replicazione dei Risultati

L'obiettivo di questo primo step è stato **verificare sperimentalmente i risultati pubblicati**, seguendo la stessa pipeline descritta nel paper.

###  Classe replicata: *MESANGIALE*

Per la classe **MESANGIALE**, sono stati utilizzati gli stessi file `.csv` presenti nel paper. Questi file sono stati salvati nella cartella:

### 📁 Contenuto della cartella `Files_old_Pollo`

| File                    | Descrizione                                  |
|-------------------------|----------------------------------------------|
| `train_mesangiale.csv` | Dataset di training per la classe MESANGIALE |
| `val_mesangiale.csv`   | Dataset di validazione                       |
| `test_mesangiale.csv`  | Dataset di test     

