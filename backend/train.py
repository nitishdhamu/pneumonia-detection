import os
import json
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch import optim

from backend.config import TrainConfig
from backend.dataset import CXRDataset
from backend.model import build_model, weighted_bce_with_logits, focal_loss_with_logits
from backend.utils import set_seed, compute_all_metrics, plot_training_curves

def train_one_epoch(model, loader, optimizer, device, loss_name, pos_weight):
    """
    Single epoch training loop.
    """
    model.train()
    total, correct, loss_sum = 0, 0, 0.0

    for x, y, _, _ in tqdm(loader, leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        logits = model(x)
        if loss_name == "focal":
            loss = focal_loss_with_logits(logits, y)
        else:
            loss = weighted_bce_with_logits(
                logits, y, pos_weight.to(device) if pos_weight is not None else None
            )
        loss.backward()
        optimizer.step()

        prob = torch.sigmoid(logits).squeeze(1)
        pred = (prob >= 0.5).float()
        correct += (pred == y).sum().item()
        total += y.numel()
        loss_sum += loss.item() * y.size(0)

    return loss_sum / total if total > 0 else 0.0, (correct / total) if total > 0 else 0.0

@torch.no_grad()
def validate(model, loader, device, loss_name="bce", pos_weight=None):
    """
    Validation loop that computes metrics and validation loss.
    """
    model.eval()
    y_true, y_prob, val_loss_sum, total = [], [], 0.0, 0

    for x, y, _, _ in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        prob = torch.sigmoid(logits).squeeze(1).cpu()
        y_true += y.cpu().numpy().tolist()
        y_prob += prob.numpy().tolist()

        if loss_name == "focal":
            loss = focal_loss_with_logits(logits, y)
        else:
            loss = weighted_bce_with_logits(
                logits, y, pos_weight.to(device) if pos_weight is not None else None
            )
        val_loss_sum += loss.item() * y.size(0)
        total += y.size(0)

    metrics = compute_all_metrics(y_true, y_prob)
    metrics["val_loss"] = (val_loss_sum / total) if total > 0 else None
    return metrics

def main():
    cfg = TrainConfig()
    os.makedirs(cfg.output_dir, exist_ok=True)
    set_seed(cfg.seed)

    device = "cpu"   # CPU-only (minor project)

    # Datasets and loaders
    train_ds = CXRDataset(cfg.data_root, "train", cfg.img_size, True)
    val_ds = CXRDataset(cfg.data_root, "val", cfg.img_size, False)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    # Model
    model = build_model(cfg.model_name, 1, pretrained=True).to(device)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # (Optional) compute pos_weight if class imbalance handling desired
    pos_weight = None

    # For early stopping
    patience = cfg.early_stop_patience
    best_auc = -1.0
    patience_counter = 0
    best_path = os.path.join(cfg.output_dir, "best.ckpt")

    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(cfg.epochs):
        print(f"Epoch {epoch+1}/{cfg.epochs}")
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, device, cfg.loss, pos_weight)
        metrics = validate(model, val_loader, device, cfg.loss, pos_weight)
        val_loss = metrics.get("val_loss", None)
        val_auc = metrics.get("roc_auc", -1.0)
        val_acc = metrics.get("accuracy", 0.0)

        # Save history
        hist["train_loss"].append(tr_loss)
        hist["val_loss"].append(val_loss if val_loss is not None else 0.0)
        hist["train_acc"].append(tr_acc)
        hist["val_acc"].append(val_acc)

        print(f"  train loss: {tr_loss:.4f}  train acc: {tr_acc:.3f}")
        print(f"  val   loss: {val_loss:.4f}  val   AUC: {val_auc:.3f}  val acc: {val_acc:.3f}")

        # Check for improvement
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            # Save best model
            torch.save({"model": model.state_dict(), "cfg": vars(cfg)}, best_path)
            with open(os.path.join(cfg.output_dir, "val_metrics.json"), "w") as f:
                json.dump(metrics, f, indent=2)
            print(f"  → New best AUC: {best_auc:.4f} (model saved)")
        else:
            patience_counter += 1
            print(f"  → No improvement. patience {patience_counter}/{patience}")

        # Early stopping (autocut)
        if patience_counter >= patience:
            print(f"Early stopping triggered (no improvement for {patience} epochs).")
            break

    # After training, save history plots and final messages
    try:
        plot_training_curves(hist, cfg.output_dir)
    except Exception as e:
        print(f"Could not plot training curves: {e}")

    print(f"Training finished. Best val ROC-AUC: {best_auc:.4f}. Best model saved at: {best_path}")

if __name__ == "__main__":
    main()
