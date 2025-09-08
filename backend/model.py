import torch
from torch import nn
import torch.nn.functional as F
import torchvision

# ---- Model builder ----
def build_model(name="densenet121", num_classes=1, pretrained=True):
    name = name.lower()
    if name.startswith("resnet"):
        m = getattr(torchvision.models, name)(weights="IMAGENET1K_V1" if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif name.startswith("densenet"):
        m = getattr(torchvision.models, name)(weights="IMAGENET1K_V1" if pretrained else None)
        m.classifier = nn.Linear(m.classifier.in_features, num_classes)
    else:
        raise ValueError(name)
    return m

# ---- Losses ----
def weighted_bce_with_logits(logits, targets, pos_weight=None):
    return F.binary_cross_entropy_with_logits(
        logits, targets.unsqueeze(1), pos_weight=pos_weight)

def focal_loss_with_logits(logits, targets, alpha=0.25, gamma=2.0):
    p = torch.sigmoid(logits).clamp(1e-6,1-1e-6)
    t = targets.unsqueeze(1)
    ce = F.binary_cross_entropy(p,t,reduction="none")
    pt = t*p+(1-t)*(1-p)
    return (alpha*(1-pt)**gamma*ce).mean()

# ---- Grad-CAM ----
import numpy as np, cv2

class GradCAM:
    def __init__(self, model: nn.Module, target_layer: str):
        self.model = model.eval()
        self.gradients = None
        self.activations = None
        layer = dict([*self.model.named_modules()])[target_layer]
        layer.register_forward_hook(self._fwd)
        layer.register_full_backward_hook(self._bwd)

    def _fwd(self, m, inp, out): self.activations = out.detach()
    def _bwd(self, m, gin, gout): self.gradients = gout[0].detach()

    def __call__(self, x):
        self.model.zero_grad()
        logits = self.model(x)
        score = logits[:,0].sum(); score.backward()
        grads, acts = self.gradients, self.activations
        weights = grads.mean(dim=(2,3), keepdim=True)
        cam = (weights*acts).sum(dim=1, keepdim=True)
        cam = torch.relu(cam); cam = (cam-cam.min())/(cam.max()+1e-8)
        return cam.cpu().numpy()[0,0], torch.sigmoid(logits).item()

def overlay_cam(rgb_img, cam, alpha=0.45):
    h,w = rgb_img.shape[:2]
    heat = (cam*255).astype(np.uint8)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)[:,:,::-1]
    over = (alpha*heat+(1-alpha)*rgb_img).clip(0,255).astype(np.uint8)
    return over
