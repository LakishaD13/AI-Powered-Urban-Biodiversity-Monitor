import math
import pandas as pd
from pathlib import Path

PROJECT = Path(r"C:/Users/laksh/OneDrive/Documents/bird_project")
PRED_CSV = Path(r"C:/Users/laksh/OneDrive/Documents/bird_project/outputs/predictions.csv")
OUT_DIR = Path(r"C:/Users/laksh/OneDrive/Documents/bird_project/outputs")
OUT_DIR.mkdir(exist_ok=True)

CONF_THRESH = 0.30  # raise to 0.50 if you want stricter filtering

df = pd.read_csv(PRED_CSV)

df = df[df["confidence"] >= CONF_THRESH].copy()
if df.empty:
    raise ValueError("No predictions above confidence threshold.")

counts = df["species_pred"].value_counts()
N = int(counts.sum())
S = int(len(counts))

H = 0.0
for c in counts:
    p = c / N
    H -= p * math.log(p)

H_norm = H / math.log(S) if S > 1 else 0.0

D = 0.0
for c in counts:
    p = c / N
    D += p * p
simpson = 1 - D

score = (H_norm + simpson) / 2

# save counts
counts_df = counts.rename_axis("species").reset_index(name="count")
counts_df.to_csv(OUT_DIR / "species_counts.csv", index=False)

report = []
report.append("=== Biodiversity Metrics Report ===")
report.append(f"Confidence threshold: {CONF_THRESH}")
report.append(f"Total detections (N): {N}")
report.append(f"Number of species (S): {S}")
report.append("")
report.append(f"Shannon index (H'): {H:.4f}")
report.append(f"Normalized Shannon (H'/ln(S)): {H_norm:.4f}")
report.append(f"Simpson diversity (1 - D): {simpson:.4f}")
report.append(f"Overall biodiversity score: {score:.4f}")
report.append("")
report.append("Species counts:")
for sp, c in counts.items():
    report.append(f"  {sp}: {c}")

(OUT_DIR / "diversity_report.txt").write_text("\n".join(report), encoding="utf-8")

print("\n".join(report))
print("\nSaved outputs/species_counts.csv and outputs/diversity_report.txt")
