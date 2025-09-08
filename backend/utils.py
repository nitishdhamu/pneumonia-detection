import random, os, numpy as np, torch
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve, roc_curve,
    confusion_matrix, precision_score, recall_score, f1_score, brier_score_loss
)

# ---- Reproducibility ----
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"]=str(seed)

# ---- Metrics ----
def compute_all_metrics(y_true, y_prob, thr=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= thr).astype(int)

    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc  = average_precision_score(y_true, y_prob)
    acc     = (y_pred == y_true).mean()
    prec    = precision_score(y_true, y_pred, zero_division=0)
    rec     = recall_score(y_true, y_pred, zero_division=0)
    f1      = f1_score(y_true, y_pred, zero_division=0)
    cm      = confusion_matrix(y_true, y_pred)

    fpr, tpr, roc_thr = roc_curve(y_true, y_prob)
    precs, recs, pr_thr = precision_recall_curve(y_true, y_prob)

    return {
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "confusion_matrix": cm.tolist(),
        "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        "pr_curve": {"precision": precs.tolist(), "recall": recs.tolist()},
        "brier": float(brier_score_loss(y_true, y_prob))
    }

# ---- Plots ----
def plot_training_curves(hist, outdir):
    os.makedirs(outdir, exist_ok=True)
    plt.figure()
    plt.plot(hist["train_loss"], label="train")
    plt.plot(hist["val_loss"], label="val")
    plt.legend(); plt.title("Loss")
    plt.savefig(os.path.join(outdir,"loss.png")); plt.close()

    plt.figure()
    plt.plot(hist["train_acc"], label="train")
    plt.plot(hist["val_acc"], label="val")
    plt.legend(); plt.title("Accuracy")
    plt.savefig(os.path.join(outdir,"acc.png")); plt.close()
