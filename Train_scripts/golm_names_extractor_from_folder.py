import os
import csv
import random

def is_image(filename):
    return os.path.splitext(filename)[1].lower() in {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}

def collect_leaf_folders(root):
    leaf_folders = []
    for dirpath, dirnames, filenames in os.walk(root):
        if any(is_image(f) for f in filenames):
            leaf_folders.append(dirpath)
    return leaf_folders

def collect_images_from_folders(folders, root):
    image_paths = []
    for folder in folders:
        for f in os.listdir(folder):
            if is_image(f):
                # full_path = os.path.join(folder, f)
                # rel_path = os.path.relpath(full_path, root)
                image_paths.append([f])  # <-- racchiudilo in una lista per il CSV

    return image_paths

def save_to_csv(image_list, filename):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['FileName'])
        writer.writerows(image_list)

def main(root_folder, train_ratio=0.7, seed=42):
    random.seed(seed)

    # Step 1: trova tutte le ultime sottocartelle (leaf)
    leaf_folders = collect_leaf_folders(root_folder)

    # Step 2: shuffla e splitta le cartelle
    random.shuffle(leaf_folders)
    split_idx = int(len(leaf_folders) * train_ratio)
    train_folders = leaf_folders[:split_idx]
    test_folders = leaf_folders[split_idx:]

    # Step 3: raccogli le immagini+
    train_images = collect_images_from_folders(train_folders, root_folder)
    test_images = collect_images_from_folders(test_folders, root_folder)

    # Step 4: salva i CSV
    train_csv = f'csv_train_seed{seed}.csv'
    test_csv = f'csv_test_seed{seed}.csv'

    save_to_csv(train_images, train_csv)
    save_to_csv(test_images, test_csv)

    print(f"Train set: {len(train_images)} immagini in {train_csv}")
    print(f"Test set: {len(test_images)} immagini in {test_csv}")

# Esempio di uso
if __name__ == '__main__':
    root_folder = '/work/grana_far2023_fomo/Pollastri_Glomeruli/Glomeruli_estratti_Lv0'  # <-- cambia qui
    main(root_folder, train_ratio=0.8, seed=123)
