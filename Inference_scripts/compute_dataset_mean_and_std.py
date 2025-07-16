import numpy as np
import os
from tqdm import tqdm
from PIL import Image as PILImage


def compute_mean_and_std_values(full_paths):
    valori_pixels = [] 
    for full_path in full_paths:
        for wsi_folder in tqdm(sorted(os.listdir(full_path))):
            for image_name in sorted(os.listdir(os.path.join(full_path, wsi_folder))):
                image_path = os.path.join(full_path, wsi_folder, image_name)
                # Le immagini sono già RGB quindi questo convert non sarebbe strettamente necessario 
                image = PILImage.open(image_path).convert("RGB") 
                img_array = np.asarray(image) / 255.0  
                # Per ogni riga (che è un pixel) ho 3 valori corrispondenti dei 3 canali, ho una matrice con una riga e 3 valori per ogni pixel 
                # Da 3X3X3 ad esempio
                reshaped_array = img_array.reshape(-1, 3)
                # Quindi qui per ciascun canale faccio lo stack dei valori dei pixels
                # Faccio lo stack di tutte  
                valori_pixels.append(reshaped_array)  

    # Quindi qui per ciascun canale faccio lo stack dei valori dei pixels su cui poi calcolo la media e la varianza 
    stack_pixels = np.vstack(valori_pixels)  
    final_mean = np.mean(stack_pixels, axis=0)
    final_std = np.std(stack_pixels, axis=0)
    return final_mean, final_std
    

if __name__ == '__main__':

    folder_1 = 'HAMAMATSU'
    folder_2 = '3DHISTECH'
    root_folder_path = '/work/grana_far2023_fomo/Pollastri_Glomeruli/Glomeruli_estratti_Lv0'
    full_path_1 = os.path.join(root_folder_path, folder_1)
    full_path_2 = os.path.join(root_folder_path, folder_2)
    full_paths = [full_path_1, full_path_2]
    final_mean, final_std = compute_mean_and_std_values(full_paths) 
    print('Media RGB:', final_mean)
    print('Deviazione standard RGB:', final_std)

    # Valori per Glomeruli_estratti_Lv1
        # Media RGB: [0.13496943 0.14678506 0.13129657]
        # Deviazione standard RGB: [0.19465959 0.19976119 0.19709547]

    # Valori per Glomeruli_estratti_Lvo
        # Media RGB: [0.10321408 0.1319403  0.07907565]
        # Deviazione standard RGB: [0.16581197 0.18537317 0.16567207]

    # Quelli di test di Pollo erano questi 
        # (0.1224, 0.1224, 0.1224), (0.0851, 0.0851, 0.0851)
