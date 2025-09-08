import os, torch, numpy as np
from PIL import Image
from torchvision import transforms
from backend.model import build_model, GradCAM, overlay_cam

class Predictor:
    def __init__(self, arch, ckpt_path, img_size=224):
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}. Train first.")

        self.device = "cpu"   # ✅ CPU only
        self.model = build_model(arch, 1, pretrained=False).to(self.device)
        state = torch.load(ckpt_path, map_location=self.device)["model"]
        self.model.load_state_dict(state)
        self.model.eval()

        self.tfm = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),
        ])

        self.target_layer = "features.denseblock4" if "densenet" in arch else "layer4"
        self.cam = GradCAM(self.model, target_layer=self.target_layer)

    def predict_with_cam(self, pil_img, out_path):
        rgb = pil_img.convert("RGB")
        x = self.tfm(rgb).unsqueeze(0).to(self.device)
        cam, prob = self.cam(x)
        cam = np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize(rgb.size)) / 255.0
        over = overlay_cam(np.array(rgb), cam, alpha=0.45)
        Image.fromarray(over).save(out_path)

        label = "Pneumonia" if prob >= 0.5 else "Normal"
        conf = prob if prob >= 0.5 else 1 - prob
        return label, float(conf)
