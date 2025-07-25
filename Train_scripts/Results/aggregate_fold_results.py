import numpy as np
import pandas as pd

# Dati dei fold
results = [
    {
        "Fold": 0,
        "Commento": "Eval finale su test set dopo training su tutti i fold",
        "Esperimento": {},
        "Accuracy": 0.8275862068965517,
        "Precision": 0.8045112177059235,
        "Recall": 0.8492062818090252,
        "Fscore": 0.8262497661338827,
        "Conf_matrix": "[[TN=109 FP=26]\n                            [FN=19 TP=107]]",
        "Weights": "/work/grana_far2023_fomo/Pollastri_Glomeruli/Train_scripts/Models_retrain/fold_0/resnet18_[['PAR_REGOL_CONT']]_New_0_net.pth"
    },
    {
        "Fold": 1,
        "Commento": "Eval finale su test set dopo training su tutti i fold",
        "Esperimento": {},
        "Accuracy": 0.6811594202898551,
        "Precision": 0.7785234376829907,
        "Recall": 0.5395348586262856,
        "Fscore": 0.6373577667617041,
        "Conf_matrix": "[[TN=166 FP=33]\n                            [FN=99 TP=116]]",
        "Weights": "/work/grana_far2023_fomo/Pollastri_Glomeruli/Train_scripts/Models_retrain/fold_1/resnet18_[['PAR_REGOL_CONT']]_New_1_net.pth"
    },
    {
        "Fold": 2,
        "Commento": "Eval finale su test set dopo training su tutti i fold",
        "Esperimento": {},
        "Accuracy": 0.7627737226277372,
        "Precision": 0.6096256358488965,
        "Recall": 0.6666666276803141,
        "Fscore": 0.6368664828267766,
        "Conf_matrix": "[[TN=304 FP=73]\n                            [FN=57 TP=114]]",
        "Weights": "/work/grana_far2023_fomo/Pollastri_Glomeruli/Train_scripts/Models_retrain/fold_2/resnet18_[['PAR_REGOL_CONT']]_New_2_net.pth"
    },
    {
        "Fold": 3,
        "Commento": "Eval finale su test set dopo training su tutti i fold",
        "Esperimento": {},
        "Accuracy": 0.6717171717171717,
        "Precision": 0.2553190946129586,
        "Recall": 0.285714217687091,
        "Fscore": 0.26965787662287,
        "Conf_matrix": "[[TN=121 FP=35]\n                            [FN=30 TP=12]]",
        "Weights": "/work/grana_far2023_fomo/Pollastri_Glomeruli/Train_scripts/Models_retrain/fold_3/resnet18_[['PAR_REGOL_CONT']]_New_3_net.pth"
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
# GRAN_fine
# Media per fold: Accuracy 0.702 | Precision 0.585 | Recall 0.566 | FScore 0.565
# PARETE REGOLARE DISCONT (capillary wall) Non ho esempi positivi infatti predice sempre negativo
# PARETE REGOLARE CONTINUA
# Media per fold: Accuracy 0.735| Precision  0.612 | Recall 0.585 | FScore 0.592


