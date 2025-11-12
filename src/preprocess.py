
# src/preprocess.py
import cv2
import numpy as np

def preprocess_image(img_or_path, target_size=(64, 64), to_gray=False, normalize=True, clahe=True):
    """
    Apdoroja vaizdą prieš prognozę arba mokymą.
    Tinka tiek GTSRB duomenims, tiek atsitiktiniams Google paveikslėliams.

    - img_or_path: kelias į failą arba np.ndarray vaizdas
    - target_size: galutinis dydis (64x64)
    - to_gray: jei True, paverčia į pilką (1 kanalas)
    - normalize: jei True, reikšmes skalauja į [0, 1]
    - clahe: kontrasto pagerinimas (geriau atpažįsta ženklus įvairiomis sąlygomis)
    """

    # Nuskaitymas
    if isinstance(img_or_path, str):
        img = cv2.imread(img_or_path)
        if img is None:
            raise FileNotFoundError(f"Failas nerastas: {img_or_path}")
    else:
        img = img_or_path.copy()

    # Konvertuojam į RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Dydžio keitimas
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

    # Jei reikia – į pilką
    if to_gray:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        if clahe:
            clahe_op = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe_op.apply(gray)
        img = np.expand_dims(gray, axis=-1)

    # Normalizavimas
    img = img.astype(np.float32)
    if normalize:
        img /= 255.0

    return img