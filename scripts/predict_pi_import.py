import csv
import numpy as np
from pathlib import Path
import tensorflow as tf
from PIL import Image

PROJECT = Path(r"C:/Users/laksh/OneDrive/Documents/bird_project")
MODEL_PATH = Path(r"C:/Users/laksh/OneDrive/Documents/bird_project/models/bird_classifier.keras")
LABELS_PATH = Path(r"C:/Users/laksh/OneDrive/Documents/bird_project/models/labels.txt")

IN_DIR = Path(r"C:/Users/laksh/OneDrive/Documents/bird_project/pi_imports/images")
OUT_CSV = Path(r"C:/Users/laksh/OneDrive/Documents/bird_project/outputs/predictions.csv")
OUT_CSV.parent.mkdir(exist_ok=True)

IMG_SIZE = (192, 192)
IMG_EXTS = {".jpg", ".jpeg", ".png"}

labels = LABELS_PATH.read_text(encoding="utf-8").splitlines()
model = tf.keras.models.load_model(MODEL_PATH)

def load_image(path: Path):
    im = Image.open(path).convert("RGB").resize(IMG_SIZE)
    x = np.array(im, dtype=np.float32)
    return np.expand_dims(x, axis=0)

rows = []
for p in sorted(IN_DIR.glob("*")):
    if p.suffix.lower() not in IMG_EXTS:
        continue
    probs = model.predict(load_image(p), verbose=0)[0]
    idx = int(np.argmax(probs))
    rows.append([p.name, labels[idx], float(probs[idx])])

with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["filename", "species_pred", "confidence"])
    w.writerows(rows)

print(f"Wrote {len(rows)} predictions -> {OUT_CSV}")