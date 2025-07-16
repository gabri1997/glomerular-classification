import pandas as pd
import numpy as np
import matplotlib as plt
import glob
import os
import json
import seaborn as sns
from tqdm import tqdm

# 1. Per prima cosa voglio leggere le augmentations e creare un Dataframe globale in cui ho sia l'originale che tutte le augmentations
records = []
main_folder_path =  '/work/grana_far2023_fomo/Pollastri_Glomeruli/Inference_scripts/Augmented_images'
for wsi_folder in tqdm(os.listdir(main_folder_path), desc='WSI folders'):
    sub_path = os.path.join(main_folder_path, wsi_folder)
    for glomerulo_folder in os.listdir(os.path.join(main_folder_path, wsi_folder)):
        #print('Entering folder {} ...'.format(glomerulo_folder))
        files_path = os.path.join(sub_path, glomerulo_folder)
        #print(os.listdir(files_path))
        try:
            with open(os.path.join(files_path, 'Augmentation.json'), 'r') as file:
                data = json.load(file)
            for aug_type, values in data["Augmentations"].items():
                row = {"Glomerulo": data["Glomerulo"], "Augmentation": aug_type}
                row.update(values)
                records.append(row)
        except:
            print('No annotation file present')
df = pd.DataFrame(records)

# 2. Fatto questo devo creare una colonna aggiuntiva in cui registro quante classi sono cambiate a seconda dell'augmentation
def compare_with_original(group):
    # Qui restituisco una sola rig che è quella con le feature del glomerulo originale s
    
    original = group[group["Augmentation"] == "Original"].iloc[0]
    diffs = []
    for _, row in group.iterrows():
        diff = (row[2:] != original[2:]).sum()  # salta le prime due colonne
        diffs.append(diff)
    group["ChangedFeatures"] = diffs
    return group

# Raggruppo per glomerulo, e passo il gruppo di righe in input alla funzione 
df = df.groupby("Glomerulo").apply(compare_with_original)

features = [col for col in df.columns if col.startswith("Feature")]
change_stats = {f: 0 for f in features}

for glomerulo in df["Glomerulo"].unique():
    sub = df[df["Glomerulo"] == glomerulo]
    original = sub[sub["Augmentation"] == "Original"].iloc[0]
    for _, row in sub.iterrows():
        for feat in features:
            if row[feat] != original[feat]:
                change_stats[feat] += 1

heatmap_data = []
for aug in df["Augmentation"].unique():
    if aug == "Original":
        continue
    changed = {feat: 0 for feat in features}
    count = 0
    for glomerulo in df["Glomerulo"].unique():
        
        sub = df[df["Glomerulo"] == glomerulo]
        orig = sub[sub["Augmentation"] == "Original"].iloc[0]
        aug_row = sub[sub["Augmentation"] == aug]
        if aug_row.empty:
            continue
        aug_row = aug_row.iloc[0]
        for feat in features:
            if aug_row[feat] != orig[feat]:
                changed[feat] += 1
        count += 1
    for feat in changed:
        changed[feat] /= count  # percentuale
    heatmap_data.append(changed)

heatmap_df = pd.DataFrame(heatmap_data, index=[a for a in df["Augmentation"].unique() if a != "Original"])
sns.heatmap(heatmap_df, annot=True, cmap="YlGnBu")
plt.ylabel("Tipo di augmentation")
plt.xlabel("Feature")
plt.savefig('correlation_heatmap.png')