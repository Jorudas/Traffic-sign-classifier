
# src/preprocess.py

import cv2
import numpy as np


def crop_roi(img):
    """
    Automatinis Region Of Interest (ROI) iškirpimas.
    Suranda kelio ženklą ir iškirpia tik jį, pašalindamas foną.
    """

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Gaussian blur kad dingtų triukšmas
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive threshold (veikia net esant blogam apšvietimui)
    thresh = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )

    # Randa kontūrus
    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return img  # jei neranda — grąžina originalą

    # Pasirenkam didžiausią kontūrą (ženklas yra didžiausias objektas)
    c = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(c)

    # Minimalus dydis — saugiklis
    if w < 10 or h < 10:
        return img

    return img[y:y+h, x:x+w]


def preprocess_image(img_or_path, target_size=(224, 224),
                     normalize=True, clahe=True, crop=True):
    """
    Universalus, profesionalus preprocessing:
    - ROI cropping
    - CLAHE kontrasto gerinimas
    - normalization
    """

    # 1) Nuskaitymas
    if isinstance(img_or_path, str):
        img = cv2.imread(img_or_path)
        if img is None:
            raise FileNotFoundError(f"Failas nerastas: {img_or_path}")
    else:
        img = img_or_path.copy()

    # 2) Konvertuojam į RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 3) Automatinis ROI cropping (svarbiausia dalis!)
    if crop:
        img = crop_roi(img)

    # 4) Dydžio keitimas
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

    # 5) Kontrasto gerinimas
    if clahe:
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)

        clahe_op = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l2 = clahe_op.apply(l)

        lab = cv2.merge((l2, a, b))
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # 6) Normalizavimas
    img = img.astype(np.float32)
    if normalize:
        img /= 255.0

    return img