import numpy as np
from PIL import Image
import os


def load_image(path: str, target_size=(150, 150)) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize(target_size)
    return np.array(img, dtype=np.float32) / 255.0


def load_batch(paths: list[str], target_size=(150, 150)) -> np.ndarray:
    return np.stack([load_image(p, target_size) for p in paths])


def extract_and_cache_features(
    image_paths: list[str],
    keras_encoder,
    cache_path: str,
    batch_size: int = 32,
    target_size=(150, 150),
) -> np.ndarray:
    if os.path.exists(cache_path):
        return np.load(cache_path)
    feats = []
    for i in range(0, len(image_paths), batch_size):
        batch = load_batch(image_paths[i:i + batch_size], target_size)
        feats.append(keras_encoder.predict(batch, verbose=0))
    result = np.concatenate(feats, axis=0)
    np.save(cache_path, result)
    return result
