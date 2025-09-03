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
                    final_pred = 'SEGM'  
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

    if label == 'GLOBAL_SEGMENTAL':
        segm_ids = all_merged.loc[all_merged['Final_pred'] == 'SEGM', 'G_ID']
        print("SEGM G_ID ->")
        for gid in segm_ids:
            print(gid)

    else:
        if label != 'INTENS':
            g_ids = all_merged.loc[all_merged['Final_pred'] == 0, 'G_ID']
            print("G_ID con predizione == 0:")
            for gid in g_ids:
                print(gid)


    print('END!')   

if __name__ == '__main__':
    label = 'SEGMENTAL'
    path = '/work/grana_far2023_fomo/Pollastri_Glomeruli/Train_scripts/Results_folds_test'
  
    aggregate(path, label)