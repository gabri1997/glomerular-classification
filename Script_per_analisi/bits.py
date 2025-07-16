from PIL import Image
import os
import numpy as np
import pandas as pd

cartella = "/nas/softechict-nas-1/fpollastri/data/istologia/images"
excel_files = "/work/grana_far2023_fomo/Pollastri_Glomeruli/Train_scripts/files/Images.xlsx"
df_image = pd.read_excel(excel_files)
images = df_image["WSI"].tolist()
found_images = 0

for file in os.listdir(cartella):
    if file in images :
        found_images += 1
        if file.lower().endswith(".tiff") or file.lower().endswith(".tif"):
            percorso_file = os.path.join(cartella, file)
            with Image.open(percorso_file) as img:
                mode = img.mode  # Es. "I;16" per 16 bit, "L" per 8 bit
                bits = img.info.get("bits", "Sconosciuto")  # Alcuni TIFF contengono l'informazione diretta
                print(f"{file}: Mode={mode}, Bits={bits}")
print("Missing images are : ", found_images)

