import os
import sys
import csv
import random
import difflib
import pandas as pd
import collections
from collections import defaultdict

def transform(input_excel):
    
    """ Questa funzione prende l'excel fornito da Magistroni che contiene le annotazioni a livello di WSI, e genera il file all_labels.csv che contiene
        Nome WSI - stringa di True e False che corrispondono alla presenza o assenza della specifica feature come indicato nel file di Magistroni"""

    df = pd.read_excel(input_excel, skiprows=1)
    print(df.columns)

    
    all_labels = []

    # Cosa vorrei fare:
    # 1 - Vorrei leggere per riga i valori delle varie colonne, se c'è la x, in corrispondenza di quel indice nell'excel metto True, se non ho la x metto false
    # In posizione numero 1 anziche mettere le colorazioni come IgA posso mettere 'anticorpo'
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

    # Label di magistroni
    lbl_dict =  {
       'ID', 'INTENSITY', 'linear', 'pseudolinear', 'coarse granular',
       'fine granular', 'diffuse/segmental', 'diffuse/global',
       'focal/segmental', 'focal/global', 'mesangial',
       'continuous regular capillary wall (subendothelial)',
       'capillary wall regular discontinuous',
       'irregular capillary wall (subendothelial)',
    }
    
    map = {

        'LIN' : 'linear',
        'PSEUDOLIN' : 'pseudolinear',
        'GRAN_GROSS' : 'coarse granular',
        'GRAN_FINE' : 'fine granular',
        'GEN_SEGM' : 'diffuse/segmental',
        'GEN_DIFF' : 'diffuse/global',
        'FOC_SEGM' : 'focal/segmental',
        'FOC_GLOB' : 'focal/global',
        'MESANGIALE' : 'mesangial',
        'PAR_REGOL_CONT' : 'continuous regular capillary wall (subendothelial)',
        'PAR_REGOL_DISCONT' : 'capillary wall regular discontinuous',
        'PAR_IRREG' : 'irregular capillary wall (subendothelial)',
        'INTENS' : 'INTENSITY',
    }

    final_dictionary = {}
    for idx, row in df.iterrows():
        final_label_string = []
        id_sample = row['ID']
        intensity = row['INTENSITY']
        
        positive_features = []
        
        for col in df.columns[2:]:  # salto 'ID' e 'INTENSITY'
            if str(row[col]).strip().upper() == 'X':
                positive_features.append(col.strip())

        inv_map = {v: k for k, v in map.items()}
        mapped_features = [inv_map.get(f.strip(), f"{f}") for f in positive_features]

        for feature in mapped_features:
            final_label_string.append(flagsdic.get(feature.strip()))     

        print(f"Sample: {id_sample}")
        print(f"Intensity: {intensity}")
        print(f"Positive features: {positive_features}")
        print(f"Mapped features: {mapped_features}")
        print("-" * 40)

        vec = [False] * 18
        vec[0] = id_sample
        vec[1] = 'Anticorpo'
        vec[17] = intensity

        for idx_label in final_label_string:
            if isinstance(idx_label, int) and 0 <= idx_label < 18:
                vec[idx_label] = True

        # Stampa stringa finale come riga CSV senza [], senza spazi
        clean_row = ",".join(str(x) for x in vec)
        clean_row_features = ",".join(str(x) for x in vec[1:])
        print('Questa è la stringa finale :', clean_row)
        print('Questa è la stringa finale con solo le features : ', clean_row_features)

        all_labels.append(clean_row)

        # Se vuoi anche salvare su file CSV:
        output_file = 'all_labels.csv'
        with open(output_file, 'w') as f:
            for row in all_labels:
                f.write(row + "\n")

        # Voglio provare a ritornare un dizionario con chiave - labels
        root, ext = os.path.splitext(id_sample)
        final_dictionary[root] = []
        final_dictionary[root].append([clean_row_features])
        
    return final_dictionary

###############################################################################

def is_image(filename):
    return os.path.splitext(filename)[1].lower() in {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}

def collect_leaf_folders(root):
    leaf_folders = []
    for dirpath, dirnames, filenames in os.walk(root):
        if any(is_image(f) for f in filenames):
            leaf_folders.append(dirpath)
    return leaf_folders

def collect_images_from_folders(folders, root):
    image_paths = []
    for folder in folders:
        for f in os.listdir(folder):
            if is_image(f):
                # full_path = os.path.join(folder, f)
                # rel_path = os.path.relpath(full_path, root)
                image_paths.append([f])  # <-- racchiudilo in una lista per il CSV

    return image_paths

def collect_dict_images_from_folders(folders, root):
    images_dict = {}
    for folder in folders:
        wsi = folder.split('/')[-1]
        images_dict[wsi] = []  # inizializza la lista per ogni cartella
        for f in os.listdir(folder):
            if is_image(f):
                images_dict[wsi].append([f])  # racchiudi il nome dell'immagine in una lista
    return images_dict

def save_labels_to_csv(merge_dictionary, example_csv):
    with open(example_csv, mode='w', newline='') as cf:
        writer = csv.writer(cf)
        for k,v in merge_dictionary.items():
            writer.writerow([k,v])


def save_to_csv(image_list, filename):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(image_list)

def save_split_csvs(merge_dictionary, train_images_dict, test_images_dict, train_csv, test_csv):
    # Creo set di immagini per train e test
    train_images_set = set()
    for imgs in train_images_dict.values():
        # imgs è lista di liste [[nome_img], ...], quindi prendo imgs[0]
        train_images_set.update([img[0] for img in imgs])

    test_images_set = set()
    for imgs in test_images_dict.values():
        test_images_set.update([img[0] for img in imgs])

    with open(train_csv, mode='w', newline='') as train_file, \
         open(test_csv, mode='w', newline='') as test_file:

        train_writer = csv.writer(train_file)
        test_writer = csv.writer(test_file)

        for image_name, label in merge_dictionary.items():
            # Se label è lista o stringa con parentesi, pulisco un po'
            if isinstance(label, list):
                label = label[0]
            if isinstance(label, str):
                label = label.strip('[]')  # rimuove parentesi quadre se presenti

            if image_name in train_images_set:
                train_writer.writerow([image_name, label])
            elif image_name in test_images_set:
                test_writer.writerow([image_name, label])
            else:
                print(f"Attenzione: immagine {image_name} non trovata né in train né in test.")


def transfer_labels_from_wsi_to_glomeruli(root_folder, final_dictionary, output_excel, train_ratio=0.7, seed=42):

    random.seed(seed)

    # Step 1: trova tutte le ultime sottocartelle (leaf)
    leaf_folders = collect_leaf_folders(root_folder)

    # Step 2: shuffla e splitta le cartelle
    random.shuffle(leaf_folders)
    split_idx = int(len(leaf_folders) * train_ratio)
    train_folders = leaf_folders[:split_idx]
    test_folders = leaf_folders[split_idx:]

    # Step 3: raccogli le immagini
    all_images_dict = collect_dict_images_from_folders(leaf_folders, root_folder)
    train_images_dict = collect_dict_images_from_folders(train_folders, root_folder)
    test_images_dict = collect_dict_images_from_folders(test_folders, root_folder)

    # Ordinamento
    all_images_dict_od = collections.OrderedDict(sorted(all_images_dict.items()))
    train_images_dict_od = collections.OrderedDict(sorted(train_images_dict.items()))
    test_images_dict_od = collections.OrderedDict(sorted(test_images_dict.items()))
    final_dictionary_od = collections.OrderedDict(sorted(final_dictionary.items()))

    print("Chiavi in all_images_dict_od:", len(all_images_dict_od))
    print("Chiavi in final_dictionary_od:", len(final_dictionary_od))

    # Merge dizionari: nome immagine -> label
    merge_dictionary = {}

    for key, vals in all_images_dict_od.items():
        if key in final_dictionary_od:
            val = final_dictionary_od[key]
            # val è lista dentro lista, es. [['...']], prendo prima lista e trasformo in stringa
            label_str = ','.join(str(x) for x in val[0])  
            for v in vals:
                image_name = v[0]
                merge_dictionary[image_name] = label_str

    example_csv = f'csv_example.csv'
    save_labels_to_csv(merge_dictionary, example_csv)

    # Step 4: salva i CSV train e test con split a livello WSI
    train_csv = f'csv_train_seed{seed}.csv'
    test_csv = f'csv_test_seed{seed}.csv'

    save_split_csvs(merge_dictionary, train_images_dict_od, test_images_dict_od, train_csv, test_csv)

    print(f"Train set: {sum(len(v) for v in train_images_dict_od.values())} immagini in {train_csv}")
    print(f"Test set: {sum(len(v) for v in test_images_dict_od.values())} immagini in {test_csv}")


if __name__ == '__main__':
    root_folder = '/work/grana_far2023_fomo/Pollastri_Glomeruli/Glomeruli_estratti_Lv0'
    input_excel = '/work/grana_far2023_fomo/Pollastri_Glomeruli/Train_scripts/Excels/IF score.xlsx'
    output_excel = '/work/grana_far2023_fomo/Pollastri_Glomeruli/Train_scripts/csv_train_seed123.csv'
    
    final_dictionary = transform(input_excel)
    #transfer_labels_from_wsi_to_glomeruli(root_folder, final_dictionary, output_excel, train_ratio=0.7, seed=42)
