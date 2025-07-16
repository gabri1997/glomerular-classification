import pandas as pd
import numpy as np
import sys
import os

# 1 - Aggiunta colonna wsi
# 2 - Aggregamento WSI, Inserimento X, Sogliatura magistroni su Segmental e Global e del 50%, Pulizia features
# 3 - Per ora a mano, una volta prodotto l'excel riassuntivo, si deve comparare con quello dei ground_truth e produrre 'Final_wsi_classification_Lv0_magistroni_norm'
# 4 - Generato il file finale, sarebbe carino verificare tra i due file excel (quello con Lv1 norm Pollastri e quello Lv0 norm Magistroni) sono 87 vs 234

# Questa funzione prende un insieme di righe, cioè un gruppo
def classify_wsi(group):    
    lunghezza = len(group)
    # calcolo le percentuali
    perc_segmental = group['Segmentation'].astype(int).sum()/lunghezza
    perc_global = group['Global'].astype(int).sum()/lunghezza
    result = {}
    result['Segmentation'] = 'X' if perc_segmental > 0.3 else ""
    result['Global'] = 'X' if perc_global >= 0.7 else ""
    return pd.Series(result)

def sogliatura(excel_pth, output_path):

    # Faccio le due sogliature nei due modi separati e poi faccio il merge, anche la colonna intensity va trattata separatamente
    df = pd.read_excel(excel_pth)
    print(df.columns)
    '''['WSI', 'Glomerulo', 'Feature mesangial', 'Feature parietal',
        'Feature cont.', 'Feature irregular', 'Feature coarse', 'Feature fine',
        'Feature segmental', 'Feature global', 'Intensity', 'Segmentation',
        'Global']'''

    # Queste sono tutte le colonne tranne segmentation e global e intensity
    columns_to_modify = df.columns.difference(['WSI','Glomerulo','Segmentation','Global', 'Intensity'])
    df[columns_to_modify] = df[columns_to_modify].astype(int)
    df_wsi = df.groupby('WSI')[columns_to_modify].mean()
    df_wsi= df_wsi.applymap(lambda x: "X" if x >= 0.5 else "")

    # Colonna intensity ho bisogno solo della media e non della X
    df_wsi_intensity = df.groupby('WSI')['Intensity'].mean()

    # Adesso devo prendere le colonne global e segmental, prendere i glomeruli con intensità superiore a 1
    # se il 70% dei glomeruli con intensità > 1 sono global allora la WSI ha global se il 30% sono segmental allora la WSI ha segmental = True
    df_intensity_filtered = df[df['Intensity'] > 1]
    magistroni_classification = df_intensity_filtered.groupby('WSI').apply(classify_wsi)

    # Pulizia features
    df_wsi = df_wsi.drop(columns=['Feature parietal'])

    df_wsi = df_wsi.rename(columns={ 'WSI':'WSI',
                            'Intensity': 'intensity',
                            'Feature coarse':'coarse', 
                            'Feature fine': 'fine', 
                            'Feature segmental':'segmental',
                            'Feature global':'global',
                            'Feature mesangial': 'mesangial', 
                            'Feature cont.': 'continuous', 
                            'Feature irregular':'irregular'
                            })

    # Ordine desiderato delle colonne
    desired_order = ['coarse', 'fine', 'segmental', 'global', 'mesangial', 'continuous', 'irregular', 'Segmentation', 'Global', 'intensity']

    # Applica l'ordinamento (mantiene solo quelle colonne che esistono effettivamente)
    df_wsi = df_wsi[[col for col in desired_order if col in df_wsi.columns]]

    # Merge dei 2 DataFrame
    df_wsi = df_wsi.join(magistroni_classification, how='left')
    df_wsi = df_wsi.join(df_wsi_intensity, how='left')

    df_wsi.to_excel(output_path)

if __name__ == '__main__':
    input_excel_path_3D = '/work/grana_far2023_fomo/Pollastri_Glomeruli/Inference_scripts/Output_excels/result_Lv0_magistroni_norm_3D.xlsx'
    input_excel_path_hama = '/work/grana_far2023_fomo/Pollastri_Glomeruli/Inference_scripts/Output_excels/result_Lv0_magistroni_norm_Hama.xlsx'
    output_path_3D = '/work/grana_far2023_fomo/Pollastri_Glomeruli/Inference_scripts/Output_excels/3D_wsi_classification_Lv0_magistroni_norm.xlsx'
    output_path_HAMA = '/work/grana_far2023_fomo/Pollastri_Glomeruli/Inference_scripts/Output_excels/HAMA_wsi_classification_Lv0_magistroni_norm.xlsx'
    sogliatura(input_excel_path_hama, output_path_HAMA)


   