from openslide import open_slide
import os
import glob
import numpy as np
import json
from PIL import Image
from tqdm import tqdm

"""Questo script prende la WSI e partendo dalle coordinate trovate con YOLO croppa tutti i glomeruli"""

def get_slide_element_at_level(slide, lvl):
    num_levels = slide.level_count
    while lvl >= 0:
        if lvl < num_levels:
            return slide.level_dimensions[lvl], lvl
        lvl -= 1
    return None

def get_dimensions_for_key(json_file, key):
    
    with open(json_file, 'r') as f:
        data = json.load(f)

    key = os.path.splitext(key)[0]
    for k in data.keys():
        if os.path.splitext(k)[0] == key:
            found_key = os.path.splitext(k)[0]  
            key_1 = found_key + '.ndpi'
            key_2 = found_key + '.svs'  

            try:
                lv_0_dims = data[key_1].get("Slide_LV0_dims", None)[0]  
                lv_1_dims = data[key_1].get("Slide_LV1_dims", None)[0]
                micrometer_lv0 = data[key_1].get("Micron_per_pixel_x_LV0", None)
                micrometer_lv1 = data[key_1].get("Micron_per_pixel_x_LV1", None)
            except KeyError:
                print(f"Chiave mancante in JSON per {key_1}")
                lv_0_dims = data[key_2].get("Slide_LV0_dims", None)[0]  
                lv_1_dims = data[key_2].get("Slide_LV1_dims", None)[0]
                micrometer_lv0 = data[key_2].get("Micron_per_pixel_x_LV0", None)
                micrometer_lv1 = data[key_2].get("Micron_per_pixel_x_LV1", None)
                return lv_0_dims, lv_1_dims, micrometer_lv0, micrometer_lv1

            return lv_0_dims, lv_1_dims, micrometer_lv0, micrometer_lv1

def generate_glomeruli_crop(image_input_folder, annotation_input_folder, output_folder):

    os.makedirs(output_folder, exist_ok=True) 
    files_input_images = sorted(glob.glob(os.path.join(image_input_folder, '*.ndpi')))
    annotations_input_files = sorted(glob.glob(os.path.join(annotation_input_folder, '*.geojson')))
    wsi_level = 1

    for image_path, annotation_path in zip(files_input_images, annotations_input_files):

        idx_box = 0
        slide = open_slide(image_path)
        slide_name = os.path.splitext(os.path.basename(image_path))[0]
        # dest_path = os.path.join(output_folder, slide_name)
        # os.makedirs(dest_path, exist_ok=True)
        print(f"Slide - {slide_name} - opened successfully")

        with open(annotation_path, 'r') as gt_file:
            annotation_boxes = json.load(gt_file)  
    
        for feature in annotation_boxes['features']:
            coordinates = feature['geometry']['coordinates'][0]
            x_min_gt = int(min(c[0] for c in coordinates))
            y_min_gt = int(min(c[1] for c in coordinates))
            x_max_gt = int(max(c[0] for c in coordinates))
            y_max_gt = int(max(c[1] for c in coordinates))
            width = x_max_gt - x_min_gt
            height = y_max_gt - y_min_gt

            # Leggi direttamente la regione da LV0
            region = slide.read_region((x_min_gt, y_min_gt), 0, (width, height)).convert("RGB")

            crop_img = Image.fromarray(np.array(region))

            idx_box += 1

            crop_filename = f"{slide_name}_glomerulo_{idx_box:03d}.png"

            final_dest_path = os.path.join(output_folder, slide_name, crop_filename)

            os.makedirs(os.path.dirname(final_dest_path), exist_ok=True)

            crop_img.save(final_dest_path)

            print(f"Saved cropped region: {final_dest_path}")


if __name__ == "__main__":

    image_input_folder = '/work/grana_far2023_fomo/Data/HAMAMATSU'
    annotation_input_folder = '/work/grana_far2023_fomo/Pollastri_Glomeruli/Train_scripts/Annotations'
    output_folder = '/work/grana_far2023_fomo/Pollastri_Glomeruli/Glomeruli_estratti_Lv0/HAMAMATSU'
    info_yaml = '/work/grana_far2023_fomo/Pollastri_Glomeruli/Train_scripts/INFO_wsi_file_dictionary_ALL.yaml'
    generate_glomeruli_crop(image_input_folder, annotation_input_folder, output_folder)
  

