from tqdm import tqdm
import os 
import json
import matplotlib.pyplot as plt
import seaborn as sns

def count_glom(folder_path):
    glomeruli_counter = 0

    for folder in os.listdir(folder_path):
        sub_path = os.path.join(folder_path, folder)
        for sub_folder in os.listdir(sub_path):
            glomeruli_counter += 1

    print(f"Il numero di glomeruli in totale è : {glomeruli_counter}")

def print_missing_folders(folder_path_1, folder_path_2):
    
    for wsi_folder in zip(sorted(os.listdir(folder_path_1)), sorted(os.listdir(folder_path_2))):

        sub_folder_path_1 = os.path.join(folder_path_1, wsi_folder[0])
        sub_folder_path_2 = os.path.join(folder_path_2, wsi_folder[1])
        setA = set(os.listdir(sub_folder_path_1))
        setB = set(os.listdir(sub_folder_path_2))

        only_in_A = sorted(setA - setB)
        only_in_B = sorted(setB - setA)

        if only_in_A:
            print(f"Questi sono gli elementi che sono presenti solo in {sub_folder_path_1} : {only_in_A}")
        if only_in_B:
            print(f"Questi sono gli elementi che sono presenti solo in {sub_folder_path_2}: {only_in_B}")

if __name__ == '__main__':
    folder_path_1 = "/work/grana_far2023_fomo/Pollastri_Glomeruli/Inference_scripts/Augmented_images_Lv0_magistroni_normalized"
    folder_path_2 = "/work/grana_far2023_fomo/Pollastri_Glomeruli/Inference_scripts/Augmented_images_Lv1_magistroni_normalized"
    count_glom(folder_path_1)
    count_glom(folder_path_2)
    print_missing_folders(folder_path_1, folder_path_2)
    print("END!")