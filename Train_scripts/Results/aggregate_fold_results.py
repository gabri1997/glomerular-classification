import numpy as np
import pandas as pd

# Dati dei fold
results = [
    {
        "Fold": 0,
        "Commento": "Eval finale su test set dopo training su tutti i fold",
        "Esperimento": {},
        "Accuracy": 0.6702127659574468,
        "Precision": 0.1538461242603607,
        "Recall": 0.09090908057851357,
        "Fscore": 0.11428102876210652,
        "Conf_matrix": "[[TN=244 FP=44]\n                            [FN=80 TP=8]]",
        "Weights": "/work/grana_far2023_fomo/Pollastri_Glomeruli/Train_scripts/Models_retrain/fold_0/resnet18_[['SEGM']]_New_0_net.pth"
    },
    {
        "Fold": 1,
        "Commento": "Eval finale su test set dopo training su tutti i fold",
        "Esperimento": {},
        "Accuracy": 0.6610878661087866,
        "Precision": 0.15277775655864492,
        "Recall": 0.35483859521335637,
        "Fscore": 0.21358798386994143,
        "Conf_matrix": "[[TN=147 FP=61]\n                            [FN=20 TP=11]]",
        "Weights": "/work/grana_far2023_fomo/Pollastri_Glomeruli/Train_scripts/Models_retrain/fold_1/resnet18_[['SEGM']]_New_1_net.pth"
    },
    {
        "Fold": 2,
        "Commento": "Eval finale su test set dopo training su tutti i fold",
        "Esperimento": {},
        "Accuracy": 0.8836633663366337,
        "Precision": 0.39999920000160005,
        "Recall": 0.04347825141777143,
        "Fscore": 0.07842957327841704,
        "Conf_matrix": "[[TN=355 FP=3]\n                            [FN=44 TP=2]]",
        "Weights": "/work/grana_far2023_fomo/Pollastri_Glomeruli/Train_scripts/Models_retrain/fold_2/resnet18_[['SEGM']]_New_2_net.pth"
    },
    {
        "Fold": 3,
        "Commento": "Eval finale su test set dopo training su tutti i fold",
        "Esperimento": {},
        "Accuracy": 0.8805970149253731,
        "Precision": 0.0,
        "Recall": 0.0,
        "Fscore": 0.0,
        "Conf_matrix": "[[TN=354 FP=25]\n                            [FN=23 TP=0]]",
        "Weights": "/work/grana_far2023_fomo/Pollastri_Glomeruli/Train_scripts/Models_retrain/fold_3/resnet18_[['SEGM']]_New_3_net.pth"
    }
]

metrics_df = pd.DataFrame([{
    "Accuracy": r["Accuracy"],
    "Precision": r["Precision"],
    "Recall": r["Recall"],
    "F1-Score": r["Fscore"]
} for r in results])


def parse_conf_matrix(conf_str):
    import re
    numbers = list(map(int, re.findall(r'\d+', conf_str)))
    return np.array([[numbers[0], numbers[1]], [numbers[2], numbers[3]]])


conf_matrices = [parse_conf_matrix(r["Conf_matrix"]) for r in results]

total_cm = np.sum(conf_matrices, axis=0)

means = metrics_df.mean()
stds = metrics_df.std()

TN, FP, FN, TP = total_cm[0, 0], total_cm[0, 1], total_cm[1, 0], total_cm[1, 1]
acc = (TP + TN) / np.sum(total_cm)
prec = TP / (TP + FP)
rec = TP / (TP + FN)
f1 = 2 * (prec * rec) / (prec + rec)

print("Media per fold:\n", means)
print("\nDeviazione standard:\n", stds)
print("\nConfusion Matrix aggregata:\n", total_cm)
print("\nMetriche globali da CM aggregata:")
print(f"  Accuracy: {acc:.4f}")
print(f"  Precision: {prec:.4f}")
print(f"  Recall: {rec:.4f}")
print(f"  F1-score: {f1:.4f}")

total_cm = np.sum([np.array(r["Conf_matrix"]) for r in results], axis=0)
TN, FP, FN, TP = total_cm[0, 0], total_cm[0, 1], total_cm[1, 0], total_cm[1, 1]
acc = (TP + TN) / np.sum(total_cm)
prec = TP / (TP + FP)
rec = TP / (TP + FN)
f1 = 2 * (prec * rec) / (prec + rec)

print("Media per fold:\n", means)
print("\nDeviazione standard:\n", stds)
print("\nConfusion Matrix aggregata:\n", total_cm)
print("\nMetriche globali da CM aggregata:")
print(f"  Accuracy: {acc:.4f}")
print(f"  Precision: {prec:.4f}")
print(f"  Recall: {rec:.4f}")
print(f"  F1-score: {f1:.4f}")

# MESANGIALE
# Media per fold: Accuracy 0.692 | Precision 0.810 | Recall 0.716 | FScore 0.745
# GRAN_GROSS
# Media per fold: Accuracy 0.689 | Precision 0.732 | Recall 0.731 | FScore 0.727
# GRAN_FINE
# Media per fold: Accuracy 0.702 | Precision 0.585 | Recall 0.566 | FScore 0.565
# PARETE REGOLARE DISCONT (capillary wall) Non ho esempi positivi infatti predice sempre negativo
# PARETE REGOLARE CONTINUA
# Media per fold: Accuracy 0.735 | Precision  0.612 | Recall 0.585  | FScore 0.592
# PARETE IRREGOLARE
# Media per fold: Accuracy 0.658 | Precision  0.595 | Recall 0.632  | FScore 0.611
# GLOBALE
# Media per fold: Accuracy 0.810 | Precision  0.876 | Recall 0.908  | FScore 0.892
# SEGMENTALE
# Media per fold: Accuracy 0.788 | Precision  0.136 | Recall  0.111 | FScore  0.122


