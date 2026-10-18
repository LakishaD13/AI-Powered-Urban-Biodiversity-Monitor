import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf

PROJECT = Path(r"C:/Users/laksh/OneDrive/Documents/bird_project")
MODEL_PATH = PROJECT / "models" / "bird15_classifier.keras"
VAL_DIR = PROJECT / "datasets" / "bird15_split" / "val"
OUT_DIR = PROJECT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

IMG_SIZE = (192, 192)  # set to the size you trained with
BATCH = 16

model = tf.keras.models.load_model(MODEL_PATH)

ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH,
    shuffle=False
)

class_names = ds.class_names
num_classes = len(class_names)

y_true = []
y_pred = []

for x, y in ds:
    probs = model.predict(x, verbose=0)
    preds = np.argmax(probs, axis=1)
    y_true.extend(y.numpy().tolist())
    y_pred.extend(preds.tolist())

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Confusion matrix
cm = np.zeros((num_classes, num_classes), dtype=int)
for t, p in zip(y_true, y_pred):
    cm[t, p] += 1

cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
cm_df.to_csv(OUT_DIR / "confusion_matrix.csv")

# Per-class accuracy
per_class_acc = []
for i, name in enumerate(class_names):
    total = cm[i, :].sum()
    correct = cm[i, i]
    acc = correct / total if total else 0
    per_class_acc.append((name, total, correct, acc))

acc_df = pd.DataFrame(per_class_acc, columns=["class", "n_val", "correct", "accuracy"])
acc_df.to_csv(OUT_DIR / "per_class_accuracy.csv", index=False)

overall_acc = (y_true == y_pred).mean()

print(f"Overall val accuracy: {overall_acc:.4f}")
print("Saved:")
print(" - outputs/confusion_matrix.csv")
print(" - outputs/per_class_accuracy.csv")
