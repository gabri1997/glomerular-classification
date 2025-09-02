import numpy as np
import pandas as pd

# Dati dei fold
results = [
    {
        "Fold": 0,
        "Commento": "Eval finale su test set dopo training su tutti i fold",
        "Esperimento": {},
        "Accuracy": 0.6941489361702128,
        "Precision": 0.17073166567520348,
        "Recall": 0.07954544550619937,
        "Fscore": 0.10852277885221487,
        "Conf_matrix": "[[TN=254 FP=34]\n                         [FN=81 TP=7]]\n                      ",
        "Weights": "/work/grana_far2023_fomo/Pollastri_Glomeruli/Train_scripts/Models_retrain/Folds/[['SEGM']]/fold_0/resnet18_[['SEGM']]_New_0_net.pth"
    },
    {
        "Fold": 1,
        "Commento": "Eval finale su test set dopo training su tutti i fold",
        "Esperimento": {},
        "Accuracy": 0.6150627615062761,
        "Precision": 0.1411764539792407,
        "Recall": 0.3870966493236615,
        "Fscore": 0.2068925996575971,
        "Conf_matrix": "[[TN=135 FP=73]\n                         [FN=19 TP=12]]\n                      ",
        "Weights": "/work/grana_far2023_fomo/Pollastri_Glomeruli/Train_scripts/Models_retrain/Folds/[['SEGM']]/fold_1/resnet18_[['SEGM']]_New_1_net.pth"
    },
    {
        "Fold": 2,
        "Commento": "Eval finale su test set dopo training su tutti i fold",
        "Esperimento": {},
        "Accuracy": 0.8861386138613861,
        "Precision": 0.4999991666680556,
        "Recall": 0.06521737712665714,
        "Fscore": 0.11538252961965693,
        "Conf_matrix": "[[TN=355 FP=3]\n                         [FN=43 TP=3]]\n                      ",
        "Weights": "/work/grana_far2023_fomo/Pollastri_Glomeruli/Train_scripts/Models_retrain/Folds/[['SEGM']]/fold_2/resnet18_[['SEGM']]_New_2_net.pth"
    },
    {
        "Fold": 3,
        "Commento": "Eval finale su test set dopo training su tutti i fold",
        "Esperimento": {},
        "Accuracy": 0.8980099502487562,
        "Precision": 0.0,
        "Recall": 0.0,
        "Fscore": 0.0,
        "Conf_matrix": "[[TN=361 FP=18]\n                         [FN=23 TP=0]]\n                      ",
        "Weights": "/work/grana_far2023_fomo/Pollastri_Glomeruli/Train_scripts/Models_retrain/Folds/[['SEGM']]/fold_3/resnet18_[['SEGM']]_New_3_net.pth"
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


# MESANGIALE
# Media per fold: Accuracy 0.717 ± 0.050 | Precision 0.822 ± 0.134 | Recall 0.742  ± 0.090 | FScore 0.768 ± 0.034 
# GRAN_GROSS
# Media per fold: Accuracy 0.662 ± 0.084 | Precision 0.706 ± 0.120 | Recall 0.707  ± 0.04  | FScore 0.703 ± 0.071
# GRAN_FINE
# Media per fold: Accuracy 0.704 ± 0.041 | Precision 0.593 ± 0.104 | Recall 0.576  ± 0.164 | FScore 0.574 ± 0.111
# PARETE REGOLARE DISCONT (capillary wall) 
# Non ho esempi positivi 
# PARETE REGOLARE CONTINUA
# Media per fold: Accuracy 0.734 ± 0.056 | Precision 0.620 ± 0.202 | Recall 0.645  ± 0.151 | FScore  0.624 ± 0.162
# PARETE IRREGOLARE
# Media per fold: Accuracy 0.646 ± 0.057 | Precision 0.585 ± 0.022 | Recall 0.600  ± 0.088 | FScore  0.590 ± 0.048
# GLOBALE
# Media per fold: Accuracy 0.798 ± 0.107 | Precision 0.880 ± 0.068 | Recall 0.884  ± 0.114 | FScore  0.879 ± 0.072
# SEGMENTALE
# Media per fold: Accuracy 0.773 ± 0.140 | Precision 0.202 ± 0.211 | Recall 0.132  ± 0.172 | FScore  0.107 ± 0.084

