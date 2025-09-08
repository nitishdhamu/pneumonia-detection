import os, json
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch import optim
from backend.config import TrainConfig
from backend.dataset import CXRDataset
from backend.model import build_model, weighted_bce_with_logits, focal_loss_with_logits
from backend.utils import set_seed, compute_all_metrics, plot_training_curves

def train_one_epoch(model, loader, optimizer, device, loss_name, pos_weight):
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
        loss.backward(); optimizer.step()

        prob = torch.sigmoid(logits).squeeze(1)
        pred = (prob >= 0.5).float()
        correct += (pred == y).sum().item()
        total += y.numel()
        loss_sum += loss.item() * y.size(0)

    return loss_sum / total, correct / total

@torch.no_grad()
def validate(model, loader, device, loss_name="bce", pos_weight=None):
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
    metrics["val_loss"] = val_loss_sum / total
    return metrics

def main():
    cfg = TrainConfig()
    os.makedirs(cfg.output_dir, exist_ok=True)
    set_seed(cfg.seed)

    device = "cpu"   # ✅ CPU only

    # datasets
    train_ds = CXRDataset(cfg.data_root, "train", cfg.img_size, True)
    val_ds = CXRDataset(cfg.data_root, "val", cfg.img_size, False)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    # model
    model = build_model(cfg.model_name, 1, pretrained=True).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    pos_weight = None  # (optional: compute class imbalance if needed)

    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_auc, best_path = -1, os.path.join(cfg.output_dir, "best.ckpt")

    for epoch in range(cfg.epochs):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, device, cfg.loss, pos_weight)
        metrics = validate(model, val_loader, device, cfg.loss, pos_weight)
        val_loss, val_auc, val_acc = metrics["val_loss"], metrics["roc_auc"], metrics["accuracy"]

        hist["train_loss"].append(tr_loss)
        hist["val_loss"].append(val_loss)
        hist["train_acc"].append(tr_acc)
        hist["val_acc"].append(val_acc)

        print(f"Epoch {epoch+1}/{cfg.epochs} | loss {tr_loss:.4f} acc {tr_acc:.3f} | "
              f"val loss {val_loss:.4f} AUC {val_auc:.3f} acc {val_acc:.3f}")

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save({"model": model.state_dict(), "cfg": vars(cfg)}, best_path)
            with open(os.path.join(cfg.output_dir, "val_metrics.json"), "w") as f:
                json.dump(metrics, f, indent=2)

    plot_training_curves(hist, cfg.output_dir)
    print(f"Best val ROC-AUC {best_auc:.3f} saved at {best_path}")

if __name__ == "__main__":
    main()
