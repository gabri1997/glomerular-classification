import os
import sys
import csv
import random
import pandas as pd
import collections
from collections import defaultdict


"""
Script per generare i file CSV contenenti le etichette da associare a immagini di glomeruli estratti da WSI (Whole Slide Images).

Questo script ha come obiettivo principale quello di:
- Leggere le etichette e l'intensità da un file Excel (una riga per WSI).
- Trasferire queste etichette alle immagini di glomeruli corrispondenti.
- Raggruppare le immagini per origine (WSI) e suddividerle coerentemente in train, validation e test.
- Salvare i risultati finali in tre file CSV utilizzabili per il training di modelli deep learning.

Razionale:
WSI provenienti dalla stessa colorazione immunofluorescente possono contenere caratteristiche morfologiche e di staining molto simili.
Per evitare **data leakage** tra i set di train, validation e test, è fondamentale assegnare **tutte le immagini provenienti dalla stessa WSI** allo stesso split.
In questo modo si evita che il modello apprenda caratteristiche specifiche della colorazione che potrebbe poi ritrovare nel validation o test set, falsando i risultati.

Logica di identificazione:
Poiché le WSI hanno nomi strutturati in modo diverso a seconda del laboratorio (R22, R23, R24), viene adottata una logica di estrazione dell'ID base su misura per ciascun prefisso, in modo da poter aggregare correttamente le immagini appartenenti alla stessa WSI.

Output:
- `csv_train_seed{N}.csv`: etichette delle immagini nel training set
- `csv_val_seed{N}.csv`: etichette delle immagini nel validation set
- `csv_test_seed{N}.csv`: etichette delle immagini nel test set
- `all_labels.csv`: tutte le etichette per tutte le immagini
- `csv_example.csv`: file di esempio con l'associazione immagine → etichetta

"""


def extract_base_id(wsi_name):
    name = wsi_name.strip()

    if name.startswith("R24") or name.startswith("R23"):
        parts = name.replace(' ', '_').split('_')
        if len(parts) >= 3:
            return '_'.join(parts[:3])
        else:
            return name

    elif name.startswith("R22"):
        parts = name.split(' ')
        return parts[0] if parts else name

    else:
        return name


def transform(input_excel):
    df = pd.read_excel(input_excel, skiprows=1)
    flagsdic = {
        'LIN': 2, 'PSEUDOLIN': 3, 'GRAN_GROSS': 4, 'GRAN_FINE': 5,
        'GEN_SEGM': 6, 'GEN_DIFF': 7, 'FOC_SEGM': 8, 'FOC_GLOB': 9,
        'MESANGIALE': 10, 'PARIETALE': 11, 'PAR_REGOL_CONT': 12,
        'PAR_REGOL_DISCONT': 13, 'PAR_IRREG': 14, 'CAPS_BOW': 15,
        'POLOVASC': 16, 'INTENS': 17,
    }

    map = {
        'LIN': 'linear', 'PSEUDOLIN': 'pseudolinear', 'GRAN_GROSS': 'coarse granular',
        'GRAN_FINE': 'fine granular', 'GEN_SEGM': 'diffuse/segmental', 'GEN_DIFF': 'diffuse/global',
        'FOC_SEGM': 'focal/segmental', 'FOC_GLOB': 'focal/global', 'MESANGIALE': 'mesangial',
        'PAR_REGOL_CONT': 'continuous regular capillary wall (subendothelial)',
        'PAR_REGOL_DISCONT': 'capillary wall regular discontinuous',
        'PAR_IRREG': 'irregular capillary wall (subendothelial)', 'INTENS': 'INTENSITY'
    }

    final_dictionary = {}
    all_labels = []

    for idx, row in df.iterrows():
        id_sample = row['ID']
        intensity = row['INTENSITY']
        positive_features = [col.strip() for col in df.columns[2:] if str(row[col]).strip().upper() == 'X']
        inv_map = {v: k for k, v in map.items()}
        mapped_features = [inv_map.get(f.strip(), f) for f in positive_features]
        vec = [False] * 18
        vec[0] = id_sample
        vec[1] = 'Anticorpo'
        vec[17] = intensity
        for feature in mapped_features:
            idx_label = flagsdic.get(feature.strip())
            if isinstance(idx_label, int) and 0 <= idx_label < 18:
                vec[idx_label] = True
        clean_row_features = ",".join(str(x) for x in vec[1:])
        root, _ = os.path.splitext(id_sample)
        final_dictionary[root] = [clean_row_features.split(',')]
        all_labels.append(",".join(str(x) for x in vec))

    with open('all_labels.csv', 'w') as f:
        for row in all_labels:
            f.write(row + "\n")

    return final_dictionary


def is_image(filename):
    return os.path.splitext(filename)[1].lower() in {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}


def collect_leaf_folders(root):
    leaf_folders = []
    for dirpath, dirnames, filenames in os.walk(root):
        if any(is_image(f) for f in filenames):
            leaf_folders.append(dirpath)
    return leaf_folders


def collect_dict_images_from_folders(folders, root):
    images_dict = {}
    for folder in folders:
        wsi = os.path.basename(folder)
        images_dict[wsi] = []
        for f in os.listdir(folder):
            if is_image(f):
                images_dict[wsi].append([f])
    return images_dict


def save_labels_to_csv(merge_dictionary, example_csv):
    with open(example_csv, mode='w', newline='') as cf:
        writer = csv.writer(cf)
        for k, v in merge_dictionary.items():
            writer.writerow([k] + v)


def save_split_csvs(merge_dictionary, train_dict, val_dict, test_dict, train_csv, val_csv, test_csv):
    def get_image_set(img_dict):
        return set(img[0] for imgs in img_dict.values() for img in imgs)

    train_set = get_image_set(train_dict)
    val_set = get_image_set(val_dict)
    test_set = get_image_set(test_dict)

    with open(train_csv, 'w', newline='') as ftrain, \
         open(val_csv, 'w', newline='') as fval, \
         open(test_csv, 'w', newline='') as ftest:

        writers = {
            'train': csv.writer(ftrain),
            'val': csv.writer(fval),
            'test': csv.writer(ftest)
        }

        for image_name, label in merge_dictionary.items():

            if image_name in train_set:
                writers['train'].writerow([image_name] + label)
            elif image_name in val_set:
                writers['val'].writerow([image_name] + label)
            elif image_name in test_set:
                writers['test'].writerow([image_name] + label)
            else:
                print(f"Attenzione: immagine {image_name} non trovata in alcuno split")

def generate_2fold_splits(root_folder, final_dictionary, output, train_ratio=0.6, val_ratio=0.2, seed=42):
    random.seed(seed)
    leaf_folders = collect_leaf_folders(root_folder)

    # Raggruppa WSI per base_id
    base_id_to_folders = defaultdict(list)
    for folder in leaf_folders:
        wsi = os.path.basename(folder)
        base_id = extract_base_id(wsi)
        base_id_to_folders[base_id].append(folder)

    base_ids = list(base_id_to_folders.keys())
    random.shuffle(base_ids)

    # Dividi in 2 fold
    mid = len(base_ids) // 2
    fold1_ids = base_ids[:mid]
    fold2_ids = base_ids[mid:]

    # Crea i due split
    folds = {'fold1': fold1_ids, 'fold2': fold2_ids}

    for fold_name, fold_ids in folds.items():
        # Dividi in train/val/test
        num_total = len(fold_ids)
        num_train = int(num_total * train_ratio)
        num_val = int(num_total * val_ratio)
        train_ids = fold_ids[:num_train]
        val_ids = fold_ids[num_train:num_train + num_val]
        test_ids = fold_ids[num_train + num_val:]

      
        train_folders = [f for bid in train_ids for f in base_id_to_folders[bid]]
        val_folders = [f for bid in val_ids for f in base_id_to_folders[bid]]
        test_folders = [f for bid in test_ids for f in base_id_to_folders[bid]]

        all_images_dict = collect_dict_images_from_folders(leaf_folders, root_folder)
        train_images_dict = collect_dict_images_from_folders(train_folders, root_folder)
        val_images_dict = collect_dict_images_from_folders(val_folders, root_folder)
        test_images_dict = collect_dict_images_from_folders(test_folders, root_folder)

        merge_dictionary = {}
        for key, vals in all_images_dict.items():
            if key in final_dictionary:
                label_list = final_dictionary[key][0]
                for v in vals:
                    image_name = v[0]
                    merge_dictionary[image_name] = label_list

        save_labels_to_csv(merge_dictionary, f'csv_example_{fold_name}.csv')
        save_split_csvs(
            merge_dictionary,
            train_images_dict, val_images_dict, test_images_dict,
            f'csv_{fold_name}_train_seed{seed}.csv',
            f'csv_{fold_name}_val_seed{seed}.csv',
            f'csv_{fold_name}_test_seed{seed}.csv'
        )

        print(f"[{fold_name.upper()}] Train: {sum(len(v) for v in train_images_dict.values())} immagini")
        print(f"[{fold_name.upper()}] Val:   {sum(len(v) for v in val_images_dict.values())} immagini")
        print(f"[{fold_name.upper()}] Test:  {sum(len(v) for v in test_images_dict.values())} immagini")

def transfer_labels_from_wsi_to_glomeruli(root_folder, final_dictionary, output, train_ratio=0.6, seed=42):
    random.seed(seed)
    leaf_folders = collect_leaf_folders(root_folder)

    base_id_to_folders = defaultdict(list)
    for folder in leaf_folders:
        wsi = os.path.basename(folder)
        base_id = extract_base_id(wsi)
        base_id_to_folders[base_id].append(folder)

    base_ids = list(base_id_to_folders.keys())
    random.shuffle(base_ids)

    num_total = len(base_ids)
    num_train = int(num_total * train_ratio)
    num_val = int(num_total * 0.15)
    train_ids = base_ids[:num_train]
    val_ids = base_ids[num_train:num_train + num_val]
    test_ids = base_ids[num_train + num_val:]

    train_folders = [f for bid in train_ids for f in base_id_to_folders[bid]]
    val_folders = [f for bid in val_ids for f in base_id_to_folders[bid]]
    test_folders = [f for bid in test_ids for f in base_id_to_folders[bid]]

    all_images_dict = collect_dict_images_from_folders(leaf_folders, root_folder)
    train_images_dict = collect_dict_images_from_folders(train_folders, root_folder)
    val_images_dict = collect_dict_images_from_folders(val_folders, root_folder)
    test_images_dict = collect_dict_images_from_folders(test_folders, root_folder)

    merge_dictionary = {}
    for key, vals in all_images_dict.items():
        matched = False
        for label_key in final_dictionary:
            if key.startswith(label_key):  # Match più flessibile
                label_list = final_dictionary[label_key][0]
                for v in vals:
                    image_name = v[0]
                    merge_dictionary[image_name] = label_list
                matched = True
                break
        if not matched:
            print(f"Nessuna etichetta trovata per WSI: {key}")
  


    #save_labels_to_csv(merge_dictionary, f'csv_example.csv')
    save_split_csvs(
        merge_dictionary, train_images_dict, val_images_dict, test_images_dict,
        f'csv_train_seed{seed}.csv', f'csv_val_seed{seed}.csv', f'csv_test_seed{seed}.csv'
    )

    print(f"Train: {sum(len(v) for v in train_images_dict.values())} immagini")
    print(f"Val: {sum(len(v) for v in val_images_dict.values())} immagini")
    print(f"Test: {sum(len(v) for v in test_images_dict.values())} immagini")


if __name__ == '__main__':
    root_folder = '/work/grana_far2023_fomo/Pollastri_Glomeruli/Glomeruli_estratti_Lv0'
    input_excel = '/work/grana_far2023_fomo/Pollastri_Glomeruli/Train_scripts/Excels/IF score.xlsx'
    output = ''
    final_dictionary = transform(input_excel)

    # Scegli se vuoi uno split o una 2-cross-fold-validation
    """
    60% dei dati al training

    15% al validation

    25% al test"""
    
    transfer_labels_from_wsi_to_glomeruli(root_folder, final_dictionary, output, train_ratio=0.6, seed=42)

    # generate_2fold_splits(
    #     root_folder=root_folder,
    #     final_dictionary=final_dictionary,
    #     output=output,
    #     train_ratio=0.6,  # 60% train, 20% val, 20% test
    #     val_ratio=0.2,
    #     seed=42
    # )
