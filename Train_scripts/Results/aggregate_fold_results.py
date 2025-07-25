import numpy as np
import pandas as pd

# Dati dei fold
results = [
        {
        "Fold": 0,
        "Commento": "Eval finale su test set dopo training su tutti i fold",
        "Esperimento": {},
        "Accuracy": 0.7279693486590039,
        "Precision": 0.6372548394848196,
        "Recall": 0.6565655902459,
        "Fscore": 0.6467611059518624,
        "Conf_matrix": "[[TN=125 FP=37]\n                            [FN=34 TP=65]]",
        "Weights": "/work/grana_far2023_fomo/Pollastri_Glomeruli/Train_scripts/Models_retrain/fold_0/resnet18_[['PAR_IRREG']]_New_0_net.pth"
    },
    {
        "Fold": 1,
        "Commento": "Eval finale su test set dopo training su tutti i fold",
        "Esperimento": {},
        "Accuracy": 0.6690821256038647,
        "Precision": 0.5803108507610958,
        "Recall": 0.6666666269841294,
        "Fscore": 0.6204936046009489,
        "Conf_matrix": "[[TN=165 FP=81]\n                            [FN=56 TP=112]]",
        "Weights": "/work/grana_far2023_fomo/Pollastri_Glomeruli/Train_scripts/Models_retrain/fold_1/resnet18_[['PAR_IRREG']]_New_1_net.pth"
    },
    {
        "Fold": 2,
        "Commento": "Eval finale su test set dopo training su tutti i fold",
        "Esperimento": {},
        "Accuracy": 0.5748175182481752,
        "Precision": 0.5659574227252161,
        "Recall": 0.5037878597050054,
        "Fscore": 0.5330611278332167,
        "Conf_matrix": "[[TN=182 FP=102]\n                            [FN=131 TP=133]]",
        "Weights": "/work/grana_far2023_fomo/Pollastri_Glomeruli/Train_scripts/Models_retrain/fold_2/resnet18_[['PAR_IRREG']]_New_2_net.pth"
    },
    {
        "Fold": 3,
        "Commento": "Eval finale su test set dopo training su tutti i fold",
        "Esperimento": {},
        "Accuracy": 0.6616161616161617,
        "Precision": 0.5980391570549846,
        "Recall": 0.7011493446954776,
        "Fscore": 0.6454976087278098,
        "Conf_matrix": "[[TN=70 FP=41]\n                            [FN=26 TP=61]]",
        "Weights": "/work/grana_far2023_fomo/Pollastri_Glomeruli/Train_scripts/Models_retrain/fold_3/resnet18_[['PAR_IRREG']]_New_3_net.pth"
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
# Media per fold: Accuracy 0.735 | Precision  0.612 | Recall 0.585 | FScore 0.592
# PARETE IRREGOLARE
# Media per fold: Accuracy 0.658 | Precision  0.595 | Recall 0.632 | FScore 0.611



