from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFile, ImageOps
from torchvision.transforms import v2
import matplotlib

from model import CNNModel

ImageFile.LOAD_TRUNCATED_IMAGES = True   # a few dataset images are truncated

IMG_SIZE = 224
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
LABELS = {0: ("Cat", "🐱"), 1: ("Dog", "🐶")}   # ImageFolder order: cats=0, dogs=1


@dataclass
class Result:
    label: str
    emoji: str
    confidence: float      # 0.5 - 1.0, confidence in the predicted class
    p_cat: float
    p_dog: float

    @property
    def level(self) -> str:
        if self.confidence >= 0.90:
            return "High"
        if self.confidence >= 0.70:
            return "Moderate"
        return "Low"


def open_image(file) -> Image.Image:
    """Open an uploaded file, fix EXIF rotation and force RGB."""
    img = Image.open(file)
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


class Predictor:
    """Loads the trained CNN once and exposes predict() and gradcam()."""

    def __init__(self, weights_path: Path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CNNModel().to(self.device)
        state = torch.load(weights_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()
        self.n_params = sum(p.numel() for p in self.model.parameters())
        self.transform = v2.Compose([
            v2.Resize((IMG_SIZE, IMG_SIZE)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(MEAN, STD),
        ])

    def _prepare(self, image: Image.Image) -> torch.Tensor:
        return self.transform(image).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def predict(self, image: Image.Image, tta: bool = False) -> Result:
        x = self._prepare(image)
        p_dog = torch.sigmoid(self.model(x)).item()
        if tta:  # average with the horizontally flipped image
            p_flip = torch.sigmoid(self.model(torch.flip(x, dims=[3]))).item()
            p_dog = (p_dog + p_flip) / 2
        label, emoji = LABELS[int(p_dog >= 0.5)]
        return Result(label, emoji, max(p_dog, 1 - p_dog), 1 - p_dog, p_dog)

    def gradcam(self, image: Image.Image, alpha: float = 0.45) -> Image.Image:
        """Grad-CAM overlay on the last conv block: where the model looked."""
        store = {}

        def hook(_, __, out):
            out.retain_grad()
            store["act"] = out

        handle = self.model.features[14].register_forward_hook(hook)   # ReLU of last block
        try:
            x = self._prepare(image)
            with torch.enable_grad():
                logit = self.model(x)
                target = logit if logit.item() >= 0 else -logit        # explain the predicted class
                self.model.zero_grad()
                target.sum().backward()
            act = store["act"]
            weights = act.grad.mean(dim=(2, 3), keepdim=True)
            cam = torch.relu((weights * act).sum(dim=1)).squeeze().detach().cpu().numpy()
        finally:
            handle.remove()

        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize(image.size, Image.BICUBIC)
        heat = matplotlib.colormaps["inferno"](np.asarray(cam_img) / 255.0)[..., :3]
        base = np.asarray(image).astype(np.float32) / 255.0
        overlay = (1 - alpha) * base + alpha * heat
        return Image.fromarray((overlay.clip(0, 1) * 255).astype(np.uint8))
