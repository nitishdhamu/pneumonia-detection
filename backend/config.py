from dataclasses import dataclass

@dataclass
class TrainConfig:
    data_root: str = "data/kaggle_chest_xray"
    output_dir: str = "outputs"
    model_name: str = "densenet121"
    img_size: int = 224
    batch_size: int = 32
    epochs: int = 5   # reduced for minor project
    lr: float = 3e-4
    weight_decay: float = 1e-4
    loss: str = "bce"   # "bce" or "focal"
    seed: int = 42
    num_workers: int = 0   # safer for Windows
    early_stop_patience: int = 1  # stop after 1 epoch with no improvement

@dataclass
class ServeConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    model_path: str = "outputs/best.ckpt"
    arch: str = "densenet121"
    img_size: int = 224
