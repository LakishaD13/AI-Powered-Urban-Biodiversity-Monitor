import random, shutil, re
from pathlib import Path

# -------- EDIT PATHS --------
CUB_ROOT = Path(r"C:/Users/laksh/Downloads/CUB_200_2011/CUB_200_2011/images")      # contains 014.Indigo_Bunting, etc
NAB_ROOT = Path(r"C:/Users/laksh/Downloads/Nabirds")                  # contains images/, images.txt, image_class_labels.txt, classes.txt
OUT_RAW  = Path(r"C:/Users/laksh/OneDrive/Documents/bird_project/datasets/bird15_raw")
# ---------------------------

SEED = 42
random.seed(SEED)

# How many total images to pull per species into RAW (cap for speed)
MAX_PER_CLASS = 220

# CUB mapping: (cub_folder, out_folder)
CUB_MAP = [
    ("118.House_Sparrow",          "House_Sparrow"),
    ("017.Cardinal",               "Northern_Cardinal"),
    ("073.Blue_Jay",               "Blue_Jay"),
    ("133.White_throated_Sparrow", "White_throated_Sparrow"),
    ("132.White_crowned_Sparrow",  "White_crowned_Sparrow"),
    ("119.Field_Sparrow",          "Field_Sparrow"),
    ("035.Purple_Finch",           "Purple_Finch"),
    ("014.Indigo_Bunting",         "Indigo_Bunting"),
    ("136.Barn_Swallow",           "Barn_Swallow"),
]

# NABirds common-name mapping: (nabirds_class_name, out_folder)
NAB_MAP = [
    ("American Robin",         "American_Robin"),
    ("Mourning Dove",          "Mourning_Dove"),
    ("Dark-eyed Junco",        "Dark_eyed_Junco"),
    ("American Goldfinch",     "American_Goldfinch"),
    ("Black-capped Chickadee", "Black_capped_Chickadee"),
    ("Downy Woodpecker",       "Downy_Woodpecker"),
    ("House Sparrow",         "House_Sparrow"),
    ("Northern Cardinal",      "Northern_Cardinal"),
    ("Blue Jay",               "Blue_Jay"),
    ("White-throated Sparrow", "White_throated_Sparrow"),
    ("White-crowned Sparrow",  "White_crowned_Sparrow"),
    ("Field Sparrow",          "Field_Sparrow"),
    ("Purple Finch",           "Purple_Finch"),
    ("Indigo Bunting",         "Indigo_Bunting"),
    ("Barn Swallow",           "Barn_Swallow"),
]

IMG_EXTS = {".jpg", ".jpeg", ".png"}

def list_images(folder: Path):
    return [p for p in folder.glob("*") if p.suffix.lower() in IMG_EXTS and p.is_file()]

def load_nabirds(nab_root: Path):
    images_txt = nab_root / "images.txt"
    labels_txt = nab_root / "image_class_labels.txt"
    classes_txt = nab_root / "classes.txt"
    images_dir = nab_root / "images"

    if not (images_txt.exists() and labels_txt.exists() and classes_txt.exists() and images_dir.exists()):
        raise FileNotFoundError("Expected images.txt, image_class_labels.txt, classes.txt, and images/ in NAB_ROOT")

    class_id_to_name = {}
    for line in classes_txt.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line: continue
        parts = line.split()
        cid = parts[0]
        cname = " ".join(parts[1:])
        class_id_to_name[cid] = cname

    img_id_to_path = {}
    for line in images_txt.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line: continue
        parts = line.split()
        img_id = parts[0]
        rel = " ".join(parts[1:])
        img_id_to_path[img_id] = rel

    img_id_to_class = {}
    for line in labels_txt.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line: continue
        parts = line.split()
        img_id = parts[0]
        cid = parts[1]
        img_id_to_class[img_id] = cid

    return images_dir, img_id_to_path, img_id_to_class, class_id_to_name

import re

def base_species_name(cname: str) -> str:
    # "American Goldfinch (Female/Nonbreeding Male)" -> "American Goldfinch"
    return cname.split("(")[0].strip()

def normalize_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())

def nabirds_paths_for(common_name: str, nab_data):
    images_dir, img_id_to_path, img_id_to_class, class_id_to_name = nab_data

    target_norm = normalize_name(common_name)

    # match class IDs where base name equals the target species name
    matching_cids = []
    for cid, cname in class_id_to_name.items():
        base = base_species_name(cname)
        if normalize_name(base) == target_norm:
            matching_cids.append(cid)

    if not matching_cids:
        raise RuntimeError(
            f"Could not find NABirds classes for '{common_name}'. "
            f"Try checking spelling vs classes.txt base names."
        )

    out = []
    for img_id, cid in img_id_to_class.items():
        if cid in matching_cids:
            rel = img_id_to_path.get(img_id)
            if rel:
                p = images_dir / rel
                if p.exists():
                    out.append(p)

    return out


OUT_RAW.mkdir(parents=True, exist_ok=True)

# --- Copy from CUB ---
for cub_folder, out_folder in CUB_MAP:
    dst = OUT_RAW / out_folder
    dst.mkdir(parents=True, exist_ok=True)

    imgs = list_images(CUB_ROOT / cub_folder)
    random.shuffle(imgs)
    imgs = imgs[:MAX_PER_CLASS]

    for p in imgs:
        shutil.copy2(p, dst / f"cub_{p.name}")
    print(f"CUB {out_folder}: copied {len(imgs)}")

# --- Copy from NABirds ---
nab_data = load_nabirds(NAB_ROOT)
for nab_name, out_folder in NAB_MAP:
    dst = OUT_RAW / out_folder
    dst.mkdir(parents=True, exist_ok=True)

    imgs = nabirds_paths_for(nab_name, nab_data)
    random.shuffle(imgs)
    imgs = imgs[:MAX_PER_CLASS]

    for p in imgs:
        shutil.copy2(p, dst / f"nab_{p.name}")
    print(f"NAB {out_folder}: copied {len(imgs)}")

print("\nDONE. Built:", OUT_RAW)
