import csv
from pathlib import Path

ROOT = Path(r"C:/Users/laksh/OneDrive/Documents/bird_project/simulated/birds_by_species")
OUT = Path(r"C:/Users/laksh/OneDrive/Documents/bird_project/species_predictions.csv")

rows = []
for species_dir in sorted([p for p in ROOT.iterdir() if p.is_dir()]):
    species = species_dir.name
    for img in sorted(species_dir.glob("*.jpg")):
        # Fake a motion-style timestamp so your event-grouping script can work if needed
        # (unique per image index)
        rows.append([img.name, "", species, "1.0000", species, "1.0000"])

with open(OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow([
        "crop_file",
        "source_motion_prefix",
        "species_final",
        "species_confidence",
        "top1_species_raw",
        "top1_conf_raw",
    ])
    w.writerows(rows)

print("Wrote:", OUT)
print("Total rows:", len(rows))
