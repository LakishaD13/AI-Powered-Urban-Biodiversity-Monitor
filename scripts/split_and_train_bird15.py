import random, shutil
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

RAW = Path(r"C:/Users/laksh/OneDrive/Documents/bird_project/datasets/bird15_raw")
SPLIT = Path(r"C:/Users/laksh/OneDrive/Documents/bird_project/datasets/bird15_split")
MODELS = Path(r"C:/Users/laksh/OneDrive/Documents/bird_project/models")
MODELS.mkdir(exist_ok=True)

VAL_N = 30
TRAIN_CAP = 160
SEED = 42
random.seed(SEED)

IMG_EXTS = {".jpg", ".jpeg", ".png"}

# --------- SPLIT ----------
train_root = SPLIT / "train"
val_root = SPLIT / "val"

if SPLIT.exists():
    # optional: wipe old split
    shutil.rmtree(SPLIT)

for species_dir in RAW.iterdir():
    if not species_dir.is_dir():
        continue
    species = species_dir.name
    imgs = [p for p in species_dir.glob("*") if p.suffix.lower() in IMG_EXTS]
    random.shuffle(imgs)

    if len(imgs) < VAL_N + 20:
        raise RuntimeError(f"{species} has only {len(imgs)} images. Add more before splitting.")

    (train_root / species).mkdir(parents=True, exist_ok=True)
    (val_root / species).mkdir(parents=True, exist_ok=True)

    val_imgs = imgs[:VAL_N]
    train_imgs = imgs[VAL_N:VAL_N + TRAIN_CAP]

    for p in val_imgs:
        shutil.copy2(p, val_root / species / p.name)
    for p in train_imgs:
        shutil.copy2(p, train_root / species / p.name)

print("Split complete:", SPLIT)

# --------- TRAIN ----------
IMG_SIZE = (192, 192)
BATCH = 16
EPOCHS_HEAD = 6
EPOCHS_FINE = 10

train_ds = tf.keras.utils.image_dataset_from_directory(
    train_root, image_size=IMG_SIZE, batch_size=BATCH, label_mode="int", shuffle=True
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    val_root, image_size=IMG_SIZE, batch_size=BATCH, label_mode="int", shuffle=False
)

class_names = train_ds.class_names
num_classes = len(class_names)
print("Classes:", class_names)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)

augment = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomTranslation(0.08, 0.08),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2),
])


base = tf.keras.applications.EfficientNetV2B0(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights="imagenet"
)
base.trainable = False


inputs = keras.Input(shape=IMG_SIZE + (3,))
x = augment(inputs)
x = tf.keras.applications.efficientnet_v2.preprocess_input(x)
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.25)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)
model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(5e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

cb = [keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)]
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_HEAD, callbacks=cb)

# Fine-tuning
base.trainable = True
for layer in base.layers[:-20]:
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(5e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FINE, callbacks=cb)

model.save(MODELS / "bird15_classifier.keras")
(MODELS / "labels.txt").write_text("\n".join(class_names), encoding="utf-8")

print("Saved model:", MODELS / "bird15_classifier.keras")
print("Saved labels:", MODELS / "labels.txt")
