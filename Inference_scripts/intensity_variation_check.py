import os
import json
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

def plot_overlay_histograms_all_glom(all_glomeruli_data, total_glomeruli):
    if not all_glomeruli_data:
        print("Nessun dato per generare istogrammi.")
        return

    # Usa tutti i glomeruli per calcolare la distribuzione
    df = pd.DataFrame(all_glomeruli_data)
    bins = 30
    y_max = total_glomeruli  # Limite massimo dell'asse Y = totale dei glomeruli

    # Sovrapposizione degli istogrammi
    plt.figure(figsize=(8, 5))

    # Istogramma Original_Intensity
    plt.hist(df["Original_Intensity"], bins=bins, color='skyblue', edgecolor='black', alpha=0.5, label='Original Intensity')

    # Istogramma Brightness_Intensity
    plt.hist(df["Brightness_Intensity"], bins=bins, color='orange', edgecolor='black', alpha=0.5, label='Brightness Intensity')

    # Aggiungi titolo, etichette e limiti
    plt.title("Distribuzione Original Intensity vs Brightness Intensity (su tutti i glomeruli)")
    plt.xlabel("Valore di Intensità")
    plt.ylabel("Numero di Glomeruli")
    plt.ylim(0, y_max)  # Imposta il limite Y uguale al totale dei glomeruli
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Salva il grafico sovrapposto
    plt.savefig("overlay_all_glomeruli_intensity_distribution.png")
    plt.close()
    print("Salvato: overlay_all_glomeruli_intensity_distribution.png")


def check(path):
    all_json_paths = []
    for folder in os.listdir(path):
        sub_path = os.path.join(path, folder)
        for sub_folder in os.listdir(sub_path):
            json_file_path = os.path.join(sub_path, sub_folder, 'Augmentation.json')
            all_json_paths.append(json_file_path)

    num_of_total_glom = len(all_json_paths)
    num_of_changed_glom = 0
    num_missing_json_file = 0
    num_intensity_eq_3 = 0
    num_intensity_in_range = 0
    num_zero_unchanged = 0

    changed_glom_data = []
    zero_unchanged_data = []  # sLista per Excel dei glomeruli con intensità 0 invariata
    all_glomeruli_data = []  # Lista per raccogliere tutti i dati dei glomeruli

    for json_file_path in tqdm(all_json_paths, desc="Analisi glomeruli"):
        try:
            with open(json_file_path, 'r') as j_file:
                data = json.load(j_file)
        except:
            num_missing_json_file += 1
            continue

        try:
            original_intensity = data["Augmentations"]["Original"]["Intensity"]
            brightness_intensity = data["Augmentations"]["Brightness"]["Intensity"]
        except KeyError:
            continue

        # Aggiungi i dati per tutti i glomeruli
        all_glomeruli_data.append({
            "Glomerulo": json_file_path,
            "Original_Intensity": original_intensity,
            "Brightness_Intensity": brightness_intensity
        })

        if brightness_intensity == 3:
            num_intensity_eq_3 += 1

        if 0 < brightness_intensity <= 2.99:
            num_intensity_in_range += 1

        if original_intensity == 0 and brightness_intensity == 0:
            num_zero_unchanged += 1
            zero_unchanged_data.append({
                "Glomerulo": json_file_path,
                "Original_Intensity": original_intensity,
                "Brightness_Intensity": brightness_intensity
            })

        if original_intensity != brightness_intensity:
            num_of_changed_glom += 1
            changed_glom_data.append({
                "Glomerulo": json_file_path,
                "Original_Intensity": original_intensity,
                "Brightness_Intensity": brightness_intensity
            })

    return (
        num_of_changed_glom, num_of_total_glom, num_missing_json_file,
        num_intensity_eq_3, num_intensity_in_range, num_zero_unchanged,
        changed_glom_data, zero_unchanged_data, all_glomeruli_data
    )


def save_results_to_excel(changed_glom_data, zero_unchanged_data):
    # Excel 1: glomeruli con intensità cambiata
    if changed_glom_data:
        df_changed = pd.DataFrame(changed_glom_data)
        df_changed.to_excel("glomeruli_changed_intensity.xlsx", index=False)
        print("Salvato: glomeruli_changed_intensity.xlsx")

    # Excel 2: glomeruli con intensità 0 invariata
    if zero_unchanged_data:
        df_zero = pd.DataFrame(zero_unchanged_data)
        df_zero.to_excel("glomeruli_intensity_zero_unchanged.xlsx", index=False)
        print("Salvato: glomeruli_intensity_zero_unchanged.xlsx")


if __name__ == '__main__':
    path = '/work/grana_far2023_fomo/Pollastri_Glomeruli/Inference_scripts/Augmented_images'
    (
        num_of_changed_glom, num_of_total_glom, num_of_missing,
        num_intensity_eq_3, num_intensity_in_range, num_zero_unchanged,
        changed_glom_data, zero_unchanged_data, all_glomeruli_data
    ) = check(path)

    print(f'Num glom changed from original to brightness : {num_of_changed_glom}')
    print(f'Num of total glom  : {num_of_total_glom}')
    print(f'Numero di missing json : {num_of_missing}')
    print(f'Numero di glomeruli con intensità (brightness) == 3 : {num_intensity_eq_3}')
    print(f'Numero di glomeruli con intensità (brightness) tra 0 e 2.99 : {num_intensity_in_range}')
    print(f'Numero di glomeruli con intensità originale e brightness pari a 0 : {num_zero_unchanged}')

    # Salvataggio in Excel dei risultati
    #save_results_to_excel(changed_glom_data, zero_unchanged_data)

    # Istogramma sovrapposto delle distribuzioni per tutti i glomeruli
    #plot_overlay_histograms_all_glom(all_glomeruli_data, num_of_total_glom)
