import os, json, torch
from torch.utils.data import DataLoader
from backend.config import TrainConfig
from backend.dataset import CXRDataset
from backend.model import build_model
from backend.utils import compute_all_metrics, plot_training_curves

def main():
    cfg = TrainConfig()
    device = "cpu"   # ✅ CPU only
    test_ds = CXRDataset(cfg.data_root,"test",cfg.img_size,False)
    test_loader = DataLoader(test_ds,batch_size=cfg.batch_size,shuffle=False,num_workers=cfg.num_workers)

    ckpt = torch.load(os.path.join(cfg.output_dir,"best.ckpt"),map_location=device)
    model = build_model(cfg.model_name,1).to(device)
    model.load_state_dict(ckpt["model"]); model.eval()

    y_true,y_prob=[],[]
    with torch.no_grad():
        for x,y,_,_ in test_loader:
            x = x.to(device)
            p = torch.sigmoid(model(x)).squeeze(1).cpu().numpy().tolist()
            y_prob+=p; y_true+=y.numpy().tolist()

    metrics = compute_all_metrics(y_true,y_prob)
    with open(os.path.join(cfg.output_dir,"test_metrics.json"),"w") as f: json.dump(metrics,f,indent=2)
    print(json.dumps(metrics,indent=2))

if __name__=="__main__": main()
