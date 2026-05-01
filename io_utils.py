from __future__ import annotations
import logging
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional
import cv2
import numpy as np
import pandas as pd
import yaml

def load_config(path: str | Path='config.yaml') -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def get_logger(name: str='vpgen', level: int=logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        fmt = logging.Formatter('[%(asctime)s] %(levelname)s | %(message)s', datefmt='%H:%M:%S')
        h.setFormatter(fmt)
        logger.addHandler(h)
        logger.setLevel(level)
    return logger

def load_image(path: str | Path, gray: bool=False) -> np.ndarray:
    path = str(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f'Image not found: {path}')
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f'Failed to load image: {path}')
    if not gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def save_image(img: np.ndarray, path: str | Path) -> None:
    path = str(path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img)

def resize_to_match(img_a: np.ndarray, img_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = img_a.shape[:2]
    img_b_resized = cv2.resize(img_b, (w, h), interpolation=cv2.INTER_AREA)
    return (img_a, img_b_resized)

@dataclass
class ImageMeta:
    image_id: str
    role: str
    category: str
    stage: str
    generated_bucket: str = ''
    magnitude: str = ''
    condition: str = ''
    source_input_id: str = ''
    prompt: str = ''
    file_path: str = ''
    notes: str = ''

    def to_dict(self) -> dict:
        return asdict(self)

def load_metadata(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        cols = list(ImageMeta('', '', '', '').to_dict().keys())
        return pd.DataFrame(columns=cols)
    return pd.read_csv(path)

def save_metadata(df: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

def append_metadata(meta: ImageMeta, path: str | Path) -> pd.DataFrame:
    df = load_metadata(path)
    new_row = pd.DataFrame([meta.to_dict()])
    df = pd.concat([df, new_row], ignore_index=True)
    save_metadata(df, path)
    return df

def project_root() -> Path:
    return Path(__file__).resolve().parent.parent

def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
