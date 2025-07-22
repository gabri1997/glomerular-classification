import os
from PIL import Image
import numpy as np
from tqdm import tqdm

def analyze_images(root_dir):
    rgb_count = 0
    gray_count = 0

    sum_r = 0
    sum_g = 0
    sum_b = 0
    sumsq_r = 0
    sumsq_g = 0
    sumsq_b = 0
    total_pixels = 0

    png_paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f.lower().endswith('.png'):
                png_paths.append(os.path.join(dirpath, f))

    print(f"[INFO] Trovate {len(png_paths)} immagini PNG.")

    for path in tqdm(png_paths, desc="Analisi immagini"):
        try:
            img = Image.open(path).convert("RGB")
            img_np = np.array(img)

            # RGB image (forzato da convert("RGB"))
            rgb_count += 1
            r, g, b = img_np[:, :, 0], img_np[:, :, 1], img_np[:, :, 2]

            sum_r += r.sum()
            sum_g += g.sum()
            sum_b += b.sum()

            sumsq_r += (r ** 2).sum()
            sumsq_g += (g ** 2).sum()
            sumsq_b += (b ** 2).sum()

            total_pixels += r.size  # = h * w
        except Exception as e:
            print(f"[ERRORE] Immagine non leggibile: {path} – {e}")

    print("\n--- RISULTATI ---")
    print(f" Immagini RGB processate:     {rgb_count}")
    print(f" Immagini saltate (errore):   {len(png_paths) - rgb_count}")

    if rgb_count > 0 and total_pixels > 0:
        # Media (valori normalizzati in [0, 1])
        mean_r = sum_r / total_pixels / 255
        mean_g = sum_g / total_pixels / 255
        mean_b = sum_b / total_pixels / 255

        # Varianza con stabilizzazione numerica
        var_r = sumsq_r / total_pixels - (sum_r / total_pixels) ** 2
        var_g = sumsq_g / total_pixels - (sum_g / total_pixels) ** 2
        var_b = sumsq_b / total_pixels - (sum_b / total_pixels) ** 2

        std_r = np.sqrt(np.clip(var_r, 0, None)) / 255
        std_g = np.sqrt(np.clip(var_g, 0, None)) / 255
        std_b = np.sqrt(np.clip(var_b, 0, None)) / 255

        print(f"\n Media per canale (R, G, B): ({mean_r:.4f}, {mean_g:.4f}, {mean_b:.4f})")
        print(f" Std per canale   (R, G, B): ({std_r:.4f}, {std_g:.4f}, {std_b:.4f})")

        # Suggerimento
        if abs(mean_g - mean_r) < 0.01 and abs(mean_g - mean_b) < 0.01:
            print("\nℹ Le immagini RGB sembrano simili tra canali → puoi trattarle come grayscale.")
        elif mean_g > mean_r and mean_g > mean_b:
            print("\n Il canale verde è dominante → potresti usare solo il canale verde.")
        else:
            print("\n I canali sono bilanciati → meglio usare RGB completo.")
    else:
        print("\n Nessuna immagine RGB valida trovata oppure 0 pixel processati.")

# ESEMPIO DI USO
if __name__ == '__main__':
    analyze_images("/work/grana_far2023_fomo/Pollastri_Glomeruli/Glomeruli_estratti_Lv0")
