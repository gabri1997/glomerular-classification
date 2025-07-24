import numpy as np
import pandas as pd

# Dati dei fold
results = [
    {
        "Fold": 0,
        "Accuracy": 0.7394636015325671,
        "Precision": 0.6232876285419432,
        "Recall": 0.8749999158653927,
        "Fscore": 0.7279950829124113,
        "Conf_matrix": [[102, 55], [13, 91]]
    },
    {
        "Fold": 1,
        "Accuracy": 0.7729468599033816,
        "Precision": 0.8874458490283182,
        "Recall": 0.7509157234096805,
        "Fscore": 0.8134870659631593,
        "Conf_matrix": [[115, 26], [68, 205]]
    },
    {
        "Fold": 2,
        "Accuracy": 0.7062043795620438,
        "Precision": 0.8771428320816335,
        "Recall": 0.7223529241799312,
        "Fscore": 0.7922530909279474,
        "Conf_matrix": [[80, 43], [118, 307]]
    },
    {
        "Fold": 3,
        "Accuracy": 0.5505050505050505,
        "Precision": 0.8526314891966853,
        "Recall": 0.5192307359467477,
        "Fscore": 0.6454135706124958,
        "Conf_matrix": [[28, 14], [75, 81]]
    }
]

metrics_df = pd.DataFrame([{
    "Accuracy": r["Accuracy"],
    "Precision": r["Precision"],
    "Recall": r["Recall"],
    "F1-Score": r["Fscore"]
} for r in results])

means = metrics_df.mean()
stds = metrics_df.std()

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


# // Media per fold:
# //  Accuracy     0.692280
# // Precision    0.810127
# // Recall       0.716875
# // F1-Score     0.744787
# // dtype: float64

# // Deviazione standard:
# //  Accuracy     0.098366
# // Precision    0.125413
# // Recall       0.147485
# // F1-Score     0.075564