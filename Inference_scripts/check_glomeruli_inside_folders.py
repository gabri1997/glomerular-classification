import pandas as pd

# Percorsi dei file Excel
file1 = '/work/grana_far2023_fomo/Pollastri_Glomeruli/Inference_scripts/Output_excels/3D_wsi_classification_Lv0_magistroni_norm.xlsx'
file2 = '/work/grana_far2023_fomo/Pollastri_Glomeruli/Inference_scripts/Output_excels/IF score.xlsx'

# Nome della colonna da leggere in entrambi i file
colonna = 'WSI'  # Modifica con il nome esatto della colonna

# Leggi le colonne dai file Excel
df1 = pd.read_excel(file1, usecols=[colonna])
df2 = pd.read_excel(file2, usecols=[colonna])

# Converti in set per confronto più semplice
set1 = set(df1[colonna].dropna().astype(str).str.strip())
set2 = set(df2[colonna].dropna().astype(str).str.strip())

# Elementi in file1 ma non in file2
solo_in_file1 = set1 - set2

# Elementi in file2 ma non in file1
solo_in_file2 = set2 - set1

print(f"Elementi in {file1} ma non in {file2}:")
for elem in solo_in_file1:
    print(elem)

print(f"\nElementi in {file2} ma non in {file1}:")
for elem in solo_in_file2:
    print(elem)
