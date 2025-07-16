import os
import logging
import pandas as pd
from image_process import handle_input
from nefro_resnet import init_nets, NefroNet
from PIL import Image as PILImage, ImageOps, ImageEnhance
from io import BytesIO
from tqdm import tqdm
import json
import argparse

parser = argparse.ArgumentParser()
# noAug or Aug
parser.add_argument('--type', default='noAug', help='no augmentation')
args = parser.parse_args()

log_file = "inference.log"
logging.basicConfig(
    filename=log_file,
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger()

# Inizializzazione delle mie varie reti
nets = init_nets()
intensity_net = NefroNet(lbl_name='intensity')
intensity_net.load()
results_list = []

def create_destination_folders(save_path, glomerulo_name, wsi_main_folder):

    if wsi_main_folder == '3DHISTECH':
        print('Wsi main folder :', wsi_main_folder)
        dir_name = glomerulo_name.split('_')[0]
    else: 
        print('Wsi main folder :', wsi_main_folder)
        dir_name = glomerulo_name.split('-')[0]

    dir_savePath = os.path.join(save_path, dir_name)
    os.makedirs(dir_savePath, exist_ok=True)
    glomerulo_name = glomerulo_name.split(".")[0]
    glomerulo_path = os.path.join(dir_name, glomerulo_name)
    final_save_path = os.path.join(save_path, glomerulo_path)
    os.makedirs(final_save_path, exist_ok=True)
    return final_save_path

def transform_img (final_save_path, pil_image, save_images):

  
    enhancer = ImageEnhance.Brightness(pil_image)
    img_intensity_increased = enhancer.enhance(3.5)
    img_rotated = pil_image.rotate(30, expand=True)
    scale_factor = 0.5
    new_size = (int(pil_image.width * scale_factor), int(pil_image.height * scale_factor))

    # Quando faccio il resize devo scegliere una tecnica di resampling:
    # LANCZOS è la più lenta ma la più qualitativa, ottima per il downscaling
    scaled_img = pil_image.resize(new_size, PILImage.Resampling.LANCZOS)

    flipped_img = ImageOps.mirror(pil_image)

    if save_images == True:
        pil_image.save(os.path.join(final_save_path, 'original.png'))
        img_intensity_increased.save(os.path.join(final_save_path, 'luminosity_increased.png'))
        img_rotated.save(os.path.join(final_save_path, 'rotated.png'))
        scaled_img.save(os.path.join(final_save_path, 'scaled.png'))
        flipped_img.save(os.path.join(final_save_path, 'flipped.png'))

    return {
        'Original' : pil_image,
        'Brightness': img_intensity_increased, 
        'Rotation': img_rotated,
        'Scale': scaled_img, 
        'Flip': flipped_img
            }

def compute_image(image_path, glomerulo_name):
    try:
        # Ho fatto il check a parte, quindi convert non serve ma lo lascio lo stesso 
        pil_image = PILImage.open(image_path).convert("RGB")

        # Just to try
        # enhancer = ImageEnhance.Brightness(pil_image)
        # test_image = enhancer.enhance(1)
        # enhancer = ImageEnhance.Contrast(pil_image)
        # test_image = enhancer.enhance(0.1)
        # test_image.save('test_image_enhanced_contrast_value.jpg')

        # Prendo solo il canale verde e lo replico 3 volte
        #processed_img, _ = handle_input(test_image)  
        processed_img, _ = handle_input(pil_image)  
        processed_img = BytesIO(processed_img)  
    except Exception as e:
        logger.error(f"Errore nel caricamento dell'immagine {image_path}: {e}")
        return None

    # Dizionario per salvare i risultati della singola immagine
    result = {"Glomerulo": glomerulo_name}

    for net in nets:
        result[f"Feature {net.lbl_name}"] = net.compute(processed_img)

    result["Intensity"] = intensity_net.compute(processed_img)

    # TRICK PER SEGM VS GLOB
    segm_pred = nets[6].compute(processed_img)
    glob_pred = nets[7].compute(processed_img)

    if glob_pred:
        segm_pred = False
    elif result["Intensity"] > 0.7:
        segm_pred = True

    result["Segmentation"] = segm_pred
    result["Global"] = glob_pred

    logger.info(f"Processed: {image_path}")
    return result


def compute_image_augmented(save_path, image_path, glomerulo_name, wsi_main_folder, save_images):

    final_save_path = create_destination_folders(save_path, glomerulo_name, wsi_main_folder)

    result = {"Glomerulo" : glomerulo_name,
              "Augmentations" : {} }
    try:
        # Ho fatto il check a parte, quindi convert non serve ma lo lascio lo stesso 
        pil_image = PILImage.open(image_path).convert("RGB")
        transformed_img = transform_img(final_save_path, pil_image, save_images)
        
    except Exception as e:
        logger.error(f"Errore nel caricamento dell'immagine {image_path}: {e}")
        return None

    # Prendo solo il canale verde e lo replico 3 volte
    for augmentation_type, image in transformed_img.items():
        processed_img, _ = handle_input(image)  
        processed_img = BytesIO(processed_img)  
        augmentation_result = {}
        for net in nets:
            # print(f'Numero classi per la rete {net.lbl_name} = {net.num_classes}')
            augmentation_result[f"Feature {net.lbl_name}"] = net.compute(processed_img)

        augmentation_result["Intensity"] = intensity_net.compute(processed_img)

        # TRICK PER SEGM VS GLOB
        segm_pred = nets[6].compute(processed_img)
        glob_pred = nets[7].compute(processed_img)

        if glob_pred:
            segm_pred = False
        elif augmentation_result["Intensity"] > 0.7:
            segm_pred = True

        augmentation_result["Segmentation"] = segm_pred
        augmentation_result["Global"] = glob_pred

        logger.info(f"Processed: {image_path}")
        result["Augmentations"][augmentation_type] = augmentation_result

    # # Provo a percorrere la strada di salvarlo in un file json
    # glomerulo_name = glomerulo_name.split(".")[0]
    # final_save_path = os.path.join(save_path, glomerulo_name)
    # Definisco il percorso del file json che voglio avere
    json_file = os.path.join(final_save_path, 'Augmentation.json')
    try:
        with open(json_file, 'w') as augmented_file:
            json.dump(result, augmented_file, indent=4)
        print(f'Json saved for glomerulo {glomerulo_name}')

    except: 
        print('Error in saving json!')
        
    return result

def save_to_excel(results_list, excel_file):
    df = pd.DataFrame(results_list)
    df.to_excel(excel_file, index=False)
    logger.info(f"File Excel salvato: {excel_file}")
    print(f"File Excel salvato: {excel_file}")

# --- MAIN ---

# Just to test and debug
# Glomerulo di test : R23 209_2A1_C3-FITC_glomerulo_004.png
# image_path = '/work/grana_far2023_fomo/Pollastri_Glomeruli/Glomeruli_estratti/HAMAMATSU/R23 209_2A1_C3-FITC/R23 209_2A1_C3-FITC_glomerulo_004.png'
# glomerulo_name = 'R23 209_2A1_C3-FITC_glomerulo_004'
# save_path = '/work/grana_far2023_fomo/Pollastri_Glomeruli/Inference_scripts/Augmented_images'
# result = compute_image(save_path, image_path, glomerulo_name)

# Choose one main input wsi folder
#wsi_main_folder = '3DHISTECH'
wsi_main_folder = 'HAMAMATSU'
root_folder_path = '/work/grana_far2023_fomo/Pollastri_Glomeruli/Glomeruli_estratti_Lv0'
full_path = os.path.join(root_folder_path, wsi_main_folder)

if args.type == 'Aug':
    print("Augmentation case selected")
    # Augmentation
    # Nella versione corrected, mi sono accorto che i valori di normlizzazione che usavo erano diversi per canale
    # ma dovevo usare solo il valore di normalizzazione del canale verde essendo replicato per 3 volte
    save_path_augmentation = '/work/grana_far2023_fomo/Pollastri_Glomeruli/Inference_scripts/Augmented_images_Lv0_magistroni_normalized_corrected'
    # Per le mie WSI
    for wsi_folder in tqdm(sorted(os.listdir(full_path))):
        for image in sorted(os.listdir(os.path.join(full_path, wsi_folder))):
            image_path = os.path.join(full_path, wsi_folder, image)
            glomerulo_name = image  # Nome del glomerulo
            result = compute_image_augmented(save_path_augmentation, image_path, glomerulo_name, wsi_main_folder, save_images=True)
            if result:
                results_list.append(result)

else:
    # No augmentation 
    # Nella versione corrected, mi sono accorto che i valori di normlizzazione che usavo erano diversi per canale
    # ma dovevo usare solo il valore di normalizzazione del canale verde essendo replicato per 3 volte
    print('No augmentation case selected')
    excel_file = "/work/grana_far2023_fomo/Pollastri_Glomeruli/Inference_scripts/Output_excels/result_Lv0_magistroni_norm_corrected_Hama.xlsx"
    # Per le mie WSI
    for wsi_folder in tqdm(sorted(os.listdir(full_path))):
        for image in sorted(os.listdir(os.path.join(full_path, wsi_folder))):
            image_path = os.path.join(full_path, wsi_folder, image)
            glomerulo_name = image  # Nome del glomerulo
            result = compute_image(image_path, glomerulo_name)
            if result:
                results_list.append(result)
    save_to_excel(results_list,excel_file)


# Per quelle di Pollastri
# for image in sorted(os.listdir(full_path)):
#     image_path = os.path.join(full_path, image)
#     glomerulo_name = image  # Nome del glomerulo
#     result = compute_image(image_path, glomerulo_name)
#     if result:
#         results_list.append(result)



