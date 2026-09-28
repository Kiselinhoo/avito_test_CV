"""
Читаем PNG-кроп и приводим его к нормальному виду.

Что делаем:
  - если есть альфа-канал, кладём картинку на белый фон,
    иначе прозрачные места станут чёрными и модель запутается;
  - если картинка очень низкая, растягиваем её, иначе текст нечитаемый.
"""

from pathlib import Path

import cv2
import numpy as np

# Если высота кропа меньше этой — растягиваем
MIN_HEIGHT = 32


def read_image(path: Path) -> np.ndarray:
    """Открывает PNG и возвращает картинку, готовую к подаче в модель."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Не смог открыть {path}")

    # PNG с прозрачностью — накладываем на белый фон
    if img.ndim == 3 and img.shape[2] == 4:
        alpha = img[:, :, 3:4].astype(np.float32) / 255.0
        img = (img[:, :, :3].astype(np.float32) * alpha
               + 255.0 * (1.0 - alpha)).astype(np.uint8)

    # Мелкие кропы растягиваем, чтобы текст был виден
    h = img.shape[0]
    if h < MIN_HEIGHT:
        scale = MIN_HEIGHT / h
        new_w = max(1, int(round(img.shape[1] * scale)))
        img = cv2.resize(img, (new_w, MIN_HEIGHT),
                         interpolation=cv2.INTER_CUBIC)

    return img