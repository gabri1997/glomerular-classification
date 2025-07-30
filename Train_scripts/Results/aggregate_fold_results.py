import numpy as np
import pandas as pd

# Dati dei fold
results = [
      {
        "Fold": 0,
        "Commento": "Eval finale su test set dopo training su tutti i fold",
        "Esperimento": {},
        "Accuracy": 0.7394636015325671,
        "Precision": 0.7542372242171844,
        "Recall": 0.6953124456787152,
        "Fscore": 0.7235721852416576,
        "Conf_matrix": "[[TN=104 FP=29]\n                                [FN=39 TP=89]]",
        "Weights": "/work/grana_far2023_fomo/Pollastri_Glomeruli/Train_scripts/Models_retrain/fold_0/resnet18_[['GRAN_GROSS']]_New_0_net.pth"
    },
    {
        "Fold": 1,
        "Commento": "Eval finale su test set dopo training su tutti i fold",
        "Esperimento": {},
        "Accuracy": 0.5483091787439613,
        "Precision": 0.5057914862628771,
        "Recall": 0.6894736479224396,
        "Fscore": 0.5835140230863547,
        "Conf_matrix": "[[TN=96 FP=128]\n                                [FN=59 TP=131]]",
        "Weights": "/work/grana_far2023_fomo/Pollastri_Glomeruli/Train_scripts/Models_retrain/fold_1/resnet18_[['GRAN_GROSS']]_New_1_net.pth"
    },
    {
        "Fold": 2,
        "Commento": "Eval finale su test set dopo training su tutti i fold",
        "Esperimento": {},
        "Accuracy": 0.7445255474452555,
        "Precision": 0.824362582879247,
        "Recall": 0.7886178648071039,
        "Fscore": 0.8060891629824649,
        "Conf_matrix": "[[TN=117 FP=62]\n                                [FN=78 TP=291]]",
        "Weights": "/work/grana_far2023_fomo/Pollastri_Glomeruli/Train_scripts/Models_retrain/fold_2/resnet18_[['GRAN_GROSS']]_New_2_net.pth"
    },
    {
        "Fold": 3,
        "Commento": "Eval finale su test set dopo training su tutti i fold",
        "Esperimento": {},
        "Accuracy": 0.5959595959595959,
        "Precision": 0.6992480677257091,
        "Recall": 0.6992480677257091,
        "Fscore": 0.6992430677614615,
        "Conf_matrix": "[[TN=25 FP=40]\n                                [FN=40 TP=93]]",
        "Weights": "/work/grana_far2023_fomo/Pollastri_Glomeruli/Train_scripts/Models_retrain/fold_3/resnet18_[['GRAN_GROSS']]_New_3_net.pth"
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
# Media per fold: Accuracy 0.724 | Precision 0.810 | Recall 0.757 | FScore 0.776
# GRAN_GROSS
# Media per fold: Accuracy 0.657 | Precision 0.700 | Recall 0.718 | FScore 0.703
# GRAN_FINE
# Media per fold: Accuracy 0.702 | Precision 0.585 | Recall 0.566 | FScore 0.565
# PARETE REGOLARE DISCONT (capillary wall) 
# Non ho esempi positivi infatti predice sempre negativo
# PARETE REGOLARE CONTINUA
# Media per fold: Accuracy 0.735 | Precision  0.612 | Recall 0.585  | FScore  0.592
# PARETE IRREGOLARE
# Media per fold: Accuracy 0.658 | Precision  0.595 | Recall 0.632  | FScore  0.611
# GLOBALE
# Media per fold: Accuracy 0.810 | Precision  0.876 | Recall 0.908  | FScore  0.892
# SEGMENTALE
# Media per fold: Accuracy 0.788 | Precision  0.136 | Recall  0.111 | FScore  0.122
# SEGM_GLOB
# Media per fold: Accuracy 0.802 | Precision  0.875 | Recall  0.894 | FScore  0.883

