import csv
from collections import Counter

# Dizionario delle etichette e relative colonne
flagsdic = {
    'LIN': 2,
    'PSEUDOLIN': 3,
    'GRAN_GROSS': 4,
    'GRAN_FINE': 5,
    'GEN_SEGM': 6,
    'GEN_DIFF': 7,
    'FOC_SEGM': 8,
    'FOC_GLOB': 9,
    'MESANGIALE': 10,
    'PARIETALE': 11,
    'PAR_REGOL_CONT': 12,
    'PAR_REGOL_DISCONT': 13,
    'PAR_IRREG': 14,
    'CAPS_BOW': 15,
    'POLOVASC': 16,
    'INTENS': 17,
}

# Percorso al file CSV
#csv_file = "/work/grana_far2023_fomo/Pollastri_Glomeruli/Train_scripts/Files/[[\'GEN_SEGM\', \'FOC_SEGM\']]_4k_training.csv"
#csv_file = "/work/grana_far2023_fomo/Pollastri_Glomeruli/Train_scripts/Files/[['MESANGIALE']]_4k_test.csv"
csv_file = "/work/grana_far2023_fomo/Pollastri_Glomeruli/Train_scripts/Base_split_over_wsi/csv_test_seed16.csv"
# ----------- ANALISI TRUE in una specifica colonna ----------- #
label_to_check = 'MESANGIALE'  # Cambia qui per usare un'altra label
col_idx = flagsdic[label_to_check]
true_count = 0
false_count = 0

with open(csv_file, mode='r', newline='') as file:
    reader = csv.reader(file)
    header = next(reader)  # Salta intestazione

    for row in reader:
        if len(row) > col_idx:
            if row[col_idx].strip().lower() == 'true':
                true_count += 1
            else:
                false_count += 1

print(f"Numero di valori 'True' per la label '{label_to_check}' (colonna {col_idx}): {true_count}")
print(f"Numero di valori 'False' per la label '{label_to_check}' (colonna {col_idx}): {false_count}")
"""Nel mio split di training:
        Numero di valori 'True' per la label 'MESANGIALE' (colonna 10): 555
        Numero di valori 'False' per la label 'MESANGIALE' (colonna 10): 236"""
"""Nel mio split di validation:
        Numero di valori 'True' per la label 'MESANGIALE' (colonna 10): 208
        Numero di valori 'False' per la label 'MESANGIALE' (colonna 10): 110
"""

##########################################################################################################################
# Questa parte serve per vedere se ci sono contemporaneamente piu label, cioè se il problmea si presenta come multilabel, cioe
# mi dice quanto è comune che le due lesioni coesistano.

# labels_to_check = ['MESANGIALE', 'PAR_REGOL_CONT', 'PAR_REGOL_DISCONT','PAR_IRREG']  # Cambia qui per usare un'altra label
# col_idx = []
# for l in labels_to_check:
#     col_idx.append(flagsdic[l])
# both_true_count = 0
# false_count = 0

# with open(csv_file, mode='r', newline='') as file:
#     reader = csv.reader(file)
#     header = next(reader)  

#     mes_only = 0
#     par_only = 0
#     both_true = 0
#     neither = 0
#     row_counter = 0

#     for row in reader:
#         row_counter += 1
#         if len(row) > max(col_idx):
#             mes = row[col_idx[0]].strip().lower() in ['true', '1']
#             par = any(row[idx].strip().lower() in ['true', '1'] for idx in col_idx[1:])
            
#             if mes and par:
#                 both_true += 1
#             elif mes and not par:
#                 mes_only += 1
#             elif not mes and par:
#                 par_only += 1
#             else:
#                 neither += 1

#     print(f"Entrambi True (MESANGIALE + almeno 1 PARIETALE): {both_true}/{row_counter}")
#     print(f"Solo MESANGIALE: {mes_only}/{row_counter}")
#     print(f"Solo PARIETALE (almeno 1): {par_only}/{row_counter}")
#     print(f"Nessuna delle due: {neither}/{row_counter}")

"""
    # Nel suo split di training 
        Entrambi True (MESANGIALE + almeno 1 PARIETALE): 368/11250 (3.3%)
        Solo MESANGIALE: 1498/11250
        Solo PARIETALE (almeno 1): 1621/11250
        Nessuna delle due: 7763/11250
    # Nel mio split di training 
        Entrambi True (MESANGIALE + almeno 1 PARIETALE): 451/791 (50%)
        Solo MESANGIALE: 104/791
        Solo PARIETALE (almeno 1): 227/791
        Nessuna delle due: 9/791
"""
###################################################################################################################

# ----------- DISTRIBUZIONE CLASSI FLOAT PER 'INTENS' ----------- #
# intens_values = []
# intens_idx = flagsdic['INTENS']

# with open(csv_file, mode='r', newline='') as file:
#     reader = csv.reader(file)
#     header = next(reader)

#     for row in reader:
#         if len(row) > intens_idx:
#             try:
#                 val = float(row[intens_idx].strip())
#                 intens_values.append(val)
#             except ValueError:
#                 continue  # ignora righe non valide

# # Conta e stampa la distribuzione ordinata
# distribution = Counter(intens_values)
# print(f"\nDistribuzione delle classi per 'INTENS' (colonna {intens_idx}):")
# for k in sorted(distribution.keys()):
#     print(f"  Classe {k}: {distribution[k]} occorrenze")

# Combo dei dati
# PAR_REGOL_CONT rapporto 1/10
# MESANGIALE rapporto  1/3.8 NEL TRAIN
# PAR_REGOL_DISCO rapporto 1/122 (96 true,11800 false)
# PAR_IRREG rapporto 1/7,2
# LIN rapporto 1/97 (121,11775) # PSEUDOLIN rapporto 1/67 (174, 11722)
# GRAN_FINE rapporto 1/16
# GRAN_GROSS rapporto 1/9 
# GEN_DIFF rapporto 1/2,2 # FOC_GLOB rapporto 1/83 (141, 11755)
# GEN_SEGM rapporto 1/27 # FOC_SEGM rapporto 1/27

# NEGLI SPLIT CON SOLO I DATI DI POLLO
# MESANGIALE rapporto  1/5 nel train mentre 40/159 nel validation, 1/4

# Soli miei dati

# Numero di valori train 'True' per la label 'MESANGIALE' (colonna 10): 555
# Numero di valori train 'False' per la label 'MESANGIALE' (colonna 10): 236
# Numero di valori validation 'True' per la label 'MESANGIALE' (colonna 10): 208
# Numero di valori validation 'False' per la label 'MESANGIALE' (colonna 10): 110


