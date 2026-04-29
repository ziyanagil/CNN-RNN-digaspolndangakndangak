import numpy as np
from PIL import Image
import os

def load_image(path: str, target_size=(224, 224)) -> np.ndarray:
    # TODO: implementasi
    raise NotImplementedError

def load_batch(paths: list[str], target_size=(224, 224)) -> np.ndarray:
    # TODO: implementasi menggunakan load_image
    raise NotImplementedError


def extract_and_cache_features(
    image_paths: list[str],
    keras_encoder,
    cache_path: str,
    batch_size: int = 32,
    target_size=(224, 224),
) -> np.ndarray:
    # TODO: implementasi
    raise NotImplementedError
