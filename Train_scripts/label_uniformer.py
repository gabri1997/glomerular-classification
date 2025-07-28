import csv

def uniformer(csv_in_path, csv_out_path, cols_idx):
    """
    Unisce GEN_SEGM e FOC_SEGM in una nuova colonna "segmental",
    GEN_DIFF e FOC_GLOB in una nuova colonna "globale".
    Non rimuove le colonne originali.

    Questo è il dizionario delle labels, ogni indice è una colonna
    
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
        'GLOB': 18,
        'SEGM' : 19, 
    }

    """
    new_rows = []
    with open(csv_in_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        for row in reader:

            if len(row) > max(cols_idx):

                segmental = 'True' if row[cols_idx[0]] == 'True' or row[cols_idx[2]] == 'True' else 'False'
                globale = 'True' if row[cols_idx[1]] == 'True' or row[cols_idx[3]] == 'True' else 'False'
                # LE COLONNE SONO NELL'ORDINE GLOBAL E SEGMENTAL
                row += [globale, segmental]
                new_rows.append(row)

    with open(csv_out_path, mode='w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(new_rows)

    print(f"CSV scritto correttamente in: {csv_out_path}")
    return new_rows


if __name__ == '__main__':
    # csv_train_in_path = '/work/grana_far2023_fomo/Pollastri_Glomeruli/Train_scripts/Files/[[\'GLOB\', \'SEGM\']]_4k_training_old.csv'
    # csv_val_in_path = '/work/grana_far2023_fomo/Pollastri_Glomeruli/Train_scripts/Files/[[\'GLOB\', \'SEGM\']]_4k_validation_old.csv'
    # csv_test_in_path = '/work/grana_far2023_fomo/Pollastri_Glomeruli/Train_scripts/Files/[[\'GLOB\', \'SEGM\']]_4k_test_old.csv'
    # csv_train_out_path = '/work/grana_far2023_fomo/Pollastri_Glomeruli/Train_scripts/Files/[[\'GLOB\',\'SEGM\']]_4k_training.csv'
    # csv_val_out_path = '/work/grana_far2023_fomo/Pollastri_Glomeruli/Train_scripts/Files/[[\'GLOB\',\'SEGM\']]_4k_validation.csv'
    # csv_test_out_path = '/work/grana_far2023_fomo/Pollastri_Glomeruli/Train_scripts/Files/[[\'GLOB\',\'SEGM\']]_4k_test.csv'

    """
    Questo file csv_input_newdata_global è stato creato dallo script label_generator_from_excel.py a partere dal file excel 'if_score' di Magistroni, le label vengono 
    già mappate dallo script nel dizionario delle label di pollastri
    """
    csv_input_newdata_global = '/work/grana_far2023_fomo/Pollastri_Glomeruli/Train_scripts/Base_split_over_wsi/csv_example.csv'
    """
    Questo file csv di output è il file identico al file di input ma in più ho due colonne finali che corrispondono alla aggregazione di due colonne, questo file viene usato
    per creare i 4 fold e allenare il modello ResNet
    """
    csv_output_newdata_global = '/work/grana_far2023_fomo/Pollastri_Glomeruli/Train_scripts/Base_split_over_wsi/csv_with_aggregated_global_and_segmental.csv'

    """
    Questa lista contiene gli indici delle colonne che voglio unificare, nel nostro caso 
    Unisce GEN_SEGM e FOC_SEGM in una nuova colonna "segmental",
    GEN_DIFF e FOC_GLOB in una nuova colonna "globale".
    """
    cols_idx = [6, 7, 8, 9]

    uniformer(csv_input_newdata_global, csv_output_newdata_global, cols_idx)

    # Questo è per uniformare le colonne del vecchio dataset di Pollastri 
    # uniformer(csv_train_in_path, csv_train_out_path, cols_idx)
    # uniformer(csv_val_in_path, csv_val_out_path, cols_idx)
    # uniformer(csv_test_in_path, csv_test_out_path, cols_idx)
