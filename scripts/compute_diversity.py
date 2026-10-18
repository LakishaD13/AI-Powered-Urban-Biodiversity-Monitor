import math
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------------
# Paths
# -----------------------------
PROJECT = Path(r"C:/Users/laksh/OneDrive/Documents/bird_project")
IN_CSV = PROJECT / "species_predictions.csv"

OUT_COUNTS = PROJECT / "species_counts.csv"
OUT_REPORT = PROJECT / "diversity_report.txt"
OUT_PLOT = PROJECT / "species_counts.png"

EXCLUDE_UNKNOWN = True

# -----------------------------
# Load predictions
# -----------------------------
df = pd.read_csv(IN_CSV)

if "species_final" not in df.columns:
    raise ValueError("species_predictions.csv must contain a 'species_final' column")

data = df.copy()

if EXCLUDE_UNKNOWN:
    data = data[data["species_final"].str.lower() != "unknown"]

# -----------------------------
# Count species
# -----------------------------
counts = data["species_final"].value_counts().sort_values(ascending=False)
counts_df = counts.rename_axis("species").reset_index(name="count")

counts_df.to_csv(OUT_COUNTS, index=False)

N = counts.sum()     # total detections
S = len(counts)      # number of species

# -----------------------------
# Diversity calculations
# -----------------------------
def shannon_index(counts):
    H = 0.0
    for c in counts:
        p = c / N
        H -= p * math.log(p)
    return H

def simpson_diversity(counts):
    D = 0.0
    for c in counts:
        p = c / N
        D += p * p
    return 1 - D

if N == 0 or S == 0:
    raise ValueError("No valid detections found")

H = shannon_index(counts)
H_norm = H / math.log(S) if S > 1 else 0.0
simpson = simpson_diversity(counts)

biodiversity_score = (H_norm + simpson) / 2

# -----------------------------
# Write report
# -----------------------------
report = []
report.append("=== Biodiversity Metrics Report ===")
report.append(f"Total detections (N): {N}")
report.append(f"Number of species (S): {S}")
report.append("")
report.append(f"Shannon index (H'): {H:.4f}")
report.append(f"Normalized Shannon (H'/ln(S)): {H_norm:.4f}")
report.append(f"Simpson diversity (1 - D): {simpson:.4f}")
report.append(f"Overall biodiversity score: {biodiversity_score:.4f}")
report.append("")
report.append("Species counts:")

for sp, c in counts.items():
    report.append(f"  {sp}: {c}")

OUT_REPORT.write_text("\n".join(report), encoding="utf-8")

print("\n".join(report))

# -----------------------------
# Plot
# -----------------------------
plt.figure()
counts.plot(kind="bar")
plt.xlabel("Species")
plt.ylabel("Detections")
plt.title("Species Detection Counts (Simulated)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(OUT_PLOT, dpi=200)

print(f"\nWrote:")
print(f"  {OUT_COUNTS}")
print(f"  {OUT_REPORT}")
print(f"  {OUT_PLOT}")
