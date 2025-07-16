from tqdm import tqdm
import os 
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Confronto le predizioni tra normalizzazione di Pollastri e quella calcolata sul dataset di Magistroni
# Allo stesso modo confronto le predizioni della estrazione dei glomeruli al livello 0 con quelle della estrazione al livello 1
   
def create_normalization_comparison_json(origin_path, magistroni_path, json_save_path, type):
    
    final_list_of_dict = []

    for folder_o, folder_m in tqdm(zip(sorted(os.listdir(origin_path)), sorted(os.listdir(magistroni_path)))):
        sub_path_o = os.path.join(origin_path, folder_o)
        sub_path_m = os.path.join(magistroni_path, folder_m)
        for sub_folder_o, sub_folder_m in zip(sorted(os.listdir(sub_path_o)), sorted(os.listdir(sub_path_m))):
            #print("Sub_folder_o : ", os.listdir(os.path.join(sub_path_o,sub_folder_o)))
            #print("Sub_folder_m : ", os.listdir(os.path.join(sub_path_m, sub_folder_m)))
            files_path_o = os.path.join(sub_path_o, sub_folder_o)
            files_path_m = os.path.join(sub_path_m, sub_folder_m)
            aug_path_o = os.path.join(files_path_o, 'Augmentation.json')
            aug_path_m = os.path.join(files_path_m, "Augmentation.json")
            
            try :
                with open(aug_path_o, 'r') as file_o : 
                    data_o = json.load(file_o)
            except FileNotFoundError:
                print(f"File not found! : {file_o}")
                continue

            try:
                with open(aug_path_m, 'r') as file_m : 
                    data_m = json.load(file_m)
            except FileNotFoundError:
                print(f"File not found! : {file_m}")
                continue

            augmentation_original = data_o["Augmentations"]["Original"]
            augmentation_magistroni = data_m["Augmentations"]["Original"]
            glom_name = data_o["Glomerulo"]

            dictionary = {'glom' : glom_name}

            for (f_o, v_o),((f_m, v_m)) in zip(sorted(augmentation_original.items()),sorted(augmentation_magistroni.items())):

                # print(f"Feature_o : {f_o}, value_o : {v_o}")
                # print(f"Feature_m : {f_m}, value_m : {v_m}")

                # Devo creare una lista di dizionari in cui salvo il nome del glomerulo e se cambiano o no le features
                changed = 1 if v_o != v_m else 0

                change = {f'{f_o}': changed}
                dictionary.update(change)

            final_list_of_dict.append(dictionary)

    # Se vuoi salvare i risultati in un file json 
    with open(json_save_path, 'w',  encoding="utf-8") as file:
         json.dump(final_list_of_dict, file, indent=4)

    # Questa print in teoria deve restituire 1421 e qualcosa cioè il numero di glomeruli che ho
    print(f"La lista di dizionari finale è lunga {len(final_list_of_dict)}")

# Sezione di visualizzazione

# 3. Operazioni da fare
#     Conteggio per feature: Per ciascuna feature, somma tutti i 1 → ti dice quante volte quella feature è cambiata.
#     Distribuzione per glomerulo: Per ogni glomerulo, somma i 1 → ti dice quante feature sono cambiate in quel caso.
#     Media / deviazione standard: per ogni feature o glomerulo.
#     (Facoltativo) Heatmap o istogramma delle frequenze.

def visualize(json_save_path, type):

    with open(json_save_path, 'r') as j:
        j_data = json.load(j)

    count_dictionary = {
                    "Feature mesangial": 0,
                    "Feature parietal": 0,
                    "Feature cont.": 0,
                    "Feature irregular": 0,
                    "Feature coarse": 0,
                    "Feature fine": 0,
                    "Feature segmental": 0,
                    "Feature global": 0,
                    "Intensity": 0,
                    "Segmentation": 0,
                    "Global": 0 }


    for dictionary in j_data:
        if dictionary["Feature mesangial"] == 1:
            count_dictionary["Feature mesangial"] += 1
        if dictionary["Feature parietal"] == 1:
            count_dictionary["Feature parietal"] += 1
        if dictionary["Feature cont."] == 1:
            count_dictionary["Feature cont."] += 1
        if dictionary["Feature irregular"] == 1:
            count_dictionary["Feature irregular"] += 1
        if dictionary["Feature coarse"] == 1:
            count_dictionary["Feature coarse"] += 1
        if dictionary["Feature fine"] == 1:
            count_dictionary["Feature fine"] += 1
        if dictionary["Feature segmental"] == 1:
            count_dictionary["Feature segmental"] += 1
        if dictionary["Feature global"] == 1:
            count_dictionary["Feature global"] += 1
        if dictionary["Intensity"] == 1:
            count_dictionary["Intensity"] += 1
        if dictionary["Segmentation"] == 1:
            count_dictionary["Segmentation"] += 1
        if dictionary["Global"] == 1:
            count_dictionary["Global"] += 1

    # Quel dizionario ti sta dicendo quante volte ciascuna feature è cambiata tra la versione originale e quella normalizzata delle immagini.
    # print(count_dictionary)

    # Capiamo in percentuale quanti glomeruli sono cambiati
    feature_mesangial_perc = count_dictionary["Feature mesangial"]/len(j_data)*100
    feature_parietal_perc = count_dictionary["Feature parietal"]/len(j_data)*100
    feature_cont_perc = count_dictionary["Feature cont."]/len(j_data)*100
    feature_irregular_perc = count_dictionary["Feature irregular"]/len(j_data)*100
    feature_coarse_perc = count_dictionary["Feature coarse"]/len(j_data)*100
    feature_fine_perc = count_dictionary["Feature fine"]/len(j_data)*100
    feature_segmental_perc = count_dictionary["Feature segmental"]/len(j_data)*100
    feature_global_perc = count_dictionary["Feature global"]/len(j_data)*100
    feature_intensity_perc = count_dictionary["Intensity"]/len(j_data)*100
    feature_segmentation_exclusive_perc = count_dictionary["Segmentation"]/len(j_data)*100
    feature_global_exclusive_perc = count_dictionary["Global"]/len(j_data)*100

    percentage_dictionary = { "feature_mesangial_perc": feature_mesangial_perc,
                       "feature_parietal_perc" :  feature_parietal_perc,
                       "feature_cont_perc" : feature_cont_perc, 
                       "feature_irregular_perc" : feature_irregular_perc,
                       "feature_coarse_perc" : feature_coarse_perc, 
                       "feature_fine_perc" : feature_fine_perc,
                       "feature_segmental_perc" : feature_segmental_perc,
                       "feature_global_perc" : feature_global_perc,
                       "feature_intensity_perc" : feature_intensity_perc,
                       "feature_segmentation_exclusive_perction_perc" : feature_segmentation_exclusive_perc,
                       "feature_global_exclusive_perc" : feature_global_exclusive_perc

    }

    #print(percentage_dictionary)

    percentage_dictionary = dict(sorted(percentage_dictionary.items(), key=lambda item: item[1], reverse=True))

    # Plot
    if type == 'normalizzazione':
        plt.figure(figsize=(12, 6))
        sns.barplot(x=list(percentage_dictionary.keys()), y=list(percentage_dictionary.values()), palette="viridis")

        plt.xticks(rotation=45, ha='right')
        plt.ylabel("Percentuale di cambiamento (%)")
        plt.title("Variazione delle feature tra normalizzazioni")
        plt.tight_layout()
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        path = '/work/grana_far2023_fomo/Pollastri_Glomeruli/Inference_scripts/Pics_output_graphs/normalization_corrected_changes.png'
        plt.savefig(path)
    
    if type == 'estrazione':
        plt.figure(figsize=(12, 6))
        sns.barplot(x=list(percentage_dictionary.keys()), y=list(percentage_dictionary.values()), palette="viridis")

        plt.xticks(rotation=45, ha='right')
        plt.ylabel("Percentuale di cambiamento (%)")
        plt.title("Variazione delle feature tra estrazioni")
        plt.tight_layout()
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        path = '/work/grana_far2023_fomo/Pollastri_Glomeruli/Inference_scripts/Pics_output_graphs/extraction_changes.png'
        plt.savefig(path)

   

def wsi_comparison(wsi_comparison_path, json_save_path):

    wsi_names_set = set()
    wsi_dictionary_list = []

    dictionary = {"n_gloms":0,
                  "total_changes":0,
                  "glomeruli_with_changes":0}
    # {
    #     "R22-42": {
    #         "n_gloms": 10,
    #         "total_changes": 23,
    #         "glomeruli_with_changes": 8
    #     }

    # }

    with open(json_save_path, 'r') as file:
        data = json.load(file)

    for glomerulo in data:
        name = glomerulo["glom"] # R22-171 IgG_glomerulo_001.png
        # Posso splittarle in '_glomerulo' e prendere quello che ho prima
        wsi_name = name.split('_glomerulo')
        wsi_name = wsi_name[0]
        wsi_names_set.add(wsi_name)
        
    print(wsi_names_set)
    print(len(wsi_names_set))
        
if __name__ == '__main__':

    # Per normalizzazione applicata nella versione Magistroni e Pollastri al livello 1
    origin_path = "/work/grana_far2023_fomo/Pollastri_Glomeruli/Inference_scripts/Augmented_images_Lv0_magistroni_normalized"
    magistroni_path = "/work/grana_far2023_fomo/Pollastri_Glomeruli/Inference_scripts/Augmented_images_Lv0_magistroni_normalized_corrected"
    json_save_path = "/work/grana_far2023_fomo/Pollastri_Glomeruli/Normalization_corrected_comparison.json"
    wsi_comparison_path = "/work/grana_far2023_fomo/Pollastri_Glomeruli/Inference_scripts/Pics_output_graphs"
    create_normalization_comparison_json(origin_path, magistroni_path, json_save_path, type = 'normalizzazione')
    visualize(json_save_path, type = 'normalizzazione')
    wsi_comparison(wsi_comparison_path, json_save_path)

    # Barplot: media cambiamenti per WSI → mostra quali WSI cambiano di più.

    # Per estrazione dei glomeruli dal livello 0 al livello 1
    # origin_path = "/work/grana_far2023_fomo/Pollastri_Glomeruli/Inference_scripts/Augmented_images_Lv0_magistroni_normalized"
    # magistroni_path = "/work/grana_far2023_fomo/Pollastri_Glomeruli/Inference_scripts/Augmented_images_Lv0_magistroni_normalized_corrected"
    # json_save_path = "/work/grana_far2023_fomo/Pollastri_Glomeruli/Normalization_corrected_comparison.json"
    # create_normalization_comparison_json(origin_path, magistroni_path, json_save_path, type = 'estrazione')
    # visualize(json_save_path, type = 'estrazione')