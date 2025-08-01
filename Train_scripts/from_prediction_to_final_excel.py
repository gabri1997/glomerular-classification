import pandas as pd
from collections import defaultdict
import os
import csv
import json

def aggregate (path, label):
    
    merged_list = []
    for fold in range(4):
        input_path = os.path.join(path, label, f"fold{fold}.csv")
        output_path = os.path.join(path, label, f"fold{fold}_aggregated.csv")

        try:
            df = pd.read_csv(input_path)
            print(df.head(10))
        except FileNotFoundError:
            print('Errore nel file')

        if label == 'GLOBAL_SEGMENTAL':
        
            intensity_path = os.path.join(path, "INTENS", f"fold{fold}.csv")
            try:
                df_intens = pd.read_csv(intensity_path)
            except FileNotFoundError:
                print('Non trovo intensity file')
        
            df_intens['G_ID'] = df_intens['Glom_name'].str.extract(r"(.*)(?=_glomerulo)")
            df_intens = df_intens.sort_values(['G_ID', 'Glom_name'])
            agg_intens = df_intens.groupby('G_ID')["Prediction_['INTENS']"].apply(list).reset_index()
          
            df['G_ID'] = df['Glom_name'].str.extract(r"(.*)(?=_glomerulo)")
            df = df.sort_values(['G_ID', 'Glom_name'])
            aggregated_df = df.groupby("G_ID")["Prediction_['GLOB', 'SEGM']"].apply(list).reset_index()

            merged_df = pd.merge(
            agg_intens,        
            aggregated_df,      
            on='G_ID'
            )

            # Da indicazioni di Magistroni
            # Ora devo scorrere ciascuna riga, ciascun valore nelle colonne, se il valore corrispondente nella colonna 
            # "Prediction_['INTENS']" > 1 lo considero nel calcolo della media, se la media è superiore a 0.7 allora scrivo GLOB se no SEGM
    
            final_preds = []

            for index, row in merged_df.iterrows():
                total = 0
                for glob_segm, intens in zip(row["Prediction_['GLOB', 'SEGM']"], row["Prediction_['INTENS']"]):
                    if intens >= 1 and glob_segm == 1:
                        total += 1

                valid_indices = [i for i, intens in enumerate(row["Prediction_['INTENS']"]) if intens >= 1]
                total = sum(row["Prediction_['GLOB', 'SEGM']"][i] == 1 for i in valid_indices)
                denominator = len(valid_indices)

                if denominator == 0:
                    final_pred = 'SEGM'  # oppure fallback decision
                else:
                    final_pred = 'GLOB' if total / denominator >= 0.7 else 'SEGM'

                final_preds.append(final_pred)

            merged_df['Final_pred'] = final_preds
            merged_list.append(merged_df)
            print(f'Fold {fold} - completato.')

        else:
        
            df['G_ID'] = df['Glom_name'].str.extract(r"(.*)(?=_glomerulo)")
            agg = df.groupby('G_ID')[f"Prediction_['{label}']"].mean().round(2)
            agg = agg.reset_index()  # per lavorare più facilmente con DataFrame
            if label != 'INTENS':
                agg['Final_pred'] = (agg[f"Prediction_['{label}']"] > 0.5).astype(int)
            merged_list.append(agg)
            
    
    if merged_list:
        all_merged = pd.concat(merged_list, ignore_index=True)
        all_merged.to_csv(os.path.join(path, label, "allfolds_aggregated.csv"), index=False)
        print("Tutti i fold concatenati e salvati in allfolds_aggregated.csv")

    print(f"SEGM G_ID -> {all_merged.loc[all_merged['Final_pred']=='SEGM','G_ID'].tolist()}")

    """
    SEGM G_ID -> 
    'R22-157 C3', 
    'R22-168 C3', 
    'R23 209_2A1_KAPPA-FITC', 
    'R23_192_1A1_IgA_ind-FITC', 
    'R23_207_1A1_C3-FITC', 
    'R23_209_2A1_LAMBDA_FITC', 
    'R23_223_2A1_C1q-FITC', 
    'R23_235_2A1_IgA-FITC', 
    'R23_235_2A1_IgG-FITC', 
    'R23_235_2A1_IgM-FITC', 
    'R23_240_2A1_IgG-FITC', 
    'R24_03_1A1_C3-FITC', 
    'R24_27_2A1_IgA-FITC', 
    'R24_27_2A1_IgG-FITC', 
    'R23_222_1A1_C1q-FITC', 
    'R23_255_2A1_C3-FITC', 
    'R23_269_1A1_IGG-FITC', 
    'R23_281_2A1_C1q-FITC', 
    'R23_281_2A1_IgA-FITC', 
    'R23_281_2A1_IgM-FITC', 
    'R23_284_1A1_IGG-FITC', 
    'R24_13_2A1_C3-FITC', 
    'R24_20_1C5_C3c-FITC', 
    'R24_25_2A1_C3-FITC', 
    'R24_45_2A1_C1q-FITC', 
    'R24_45_2A1_IgA-FITC', 
    'R24_45_2A1_IgG-FITC', 
    'R24_45_2A1_IgM-FITC', 
    'R22-181 C3', 
    'R24_10_2A1_KAPPA-FITC', 
    'R24_10_2A1_LAMBDA-FITC', 
    'R23 210_1A1_C3_IND-FITC', 
    'R23 210_1A1_IgG_IND-FITC', 
    'R23_187_2A1_C3-FITC', 
    'R23_187_2A1_IgA-FITC'
    
    """
    print('END!')   

    
if __name__ == '__main__':
    label = 'INTENS'
    path = '/work/grana_far2023_fomo/Pollastri_Glomeruli/Train_scripts/Results_with_images'
  
    aggregate(path, label)