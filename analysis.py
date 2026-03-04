"""
Time Pressure, Accuracy, and Confidence Calibration
Pre-registered Analysis Script
-----------------------------------------------------

Expected CSV columns (one row per participant × question):
  participant_id   : unique participant identifier
  condition        : 'treatment' or 'control'  (or 1/0)
  question_id      : question identifier (1–5)
  response         : participant's numeric estimate (0–100)
  true_value       : correct answer (0–100)
  confidence       : self-reported confidence (1–100 scale)
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

# ── 0. Load data ────────────────────────────────────────────────────────────

# Replace this path with your actual data file
DATA_PATH = "data.csv"
df = pd.read_csv(DATA_PATH)

# Normalise condition column to 0/1
if pd.api.types.is_numeric_dtype(df["condition"]):
    df["time_pressure"] = df["condition"].astype(int)
else:
    df["time_pressure"] = (df["condition"].astype(str).str.lower() == "treatment").astype(int)

# ── 1. Derived variables ─────────────────────────────────────────────────────

# Accuracy: 1 if |response - true_value| ≤ 5, else 0
df["accuracy"] = (np.abs(df["response"] - df["true_value"]) <= 5).astype(int)

# Calibration gap (confidence already on 0–100 scale)
# CalibrationGap = Confidence - (100 × Accuracy)
# Positive gap  → overconfident; negative → underconfident
df["calibration_gap"] = df["confidence"] - (100 * df["accuracy"])

# Question fixed-effect dummies (C() handled by statsmodels)
df["question_id"] = df["question_id"].astype("category")

# ── 2. Exclusion rule ────────────────────────────────────────────────────────

q_per_participant = df.groupby("participant_id")["question_id"].count()
keep = q_per_participant[q_per_participant >= 4].index
df = df[df["participant_id"].isin(keep)].copy()

print(f"Participants retained: {df['participant_id'].nunique()}")
print(f"  Control:   {df[df['time_pressure']==0]['participant_id'].nunique()}")
print(f"  Treatment: {df[df['time_pressure']==1]['participant_id'].nunique()}")

# ── 3. Descriptive statistics ────────────────────────────────────────────────

desc = (
    df.groupby("time_pressure")[["accuracy", "confidence", "calibration_gap"]]
    .agg(["mean", "std", "count"])
)
print("\n── Descriptive Statistics ──")
print(desc.to_string())

# ── 4. H1 – Time pressure reduces accuracy ───────────────────────────────────
# OLS: accuracy_ij = α0 + α1·TimePressure_i + γ_j + u_ij
# Standard errors clustered at the participant level

model_h1 = smf.ols(
    "accuracy ~ time_pressure + C(question_id)",
    data=df
).fit(
    cov_type="cluster",
    cov_kwds={"groups": df["participant_id"]}
)

print("\n── H1: Effect of Time Pressure on Accuracy ──")
print(model_h1.summary().tables[1])
alpha1 = model_h1.params["time_pressure"]
p_h1   = model_h1.pvalues["time_pressure"]
print(f"\nα1 = {alpha1:.4f}  (p = {p_h1:.4f})")
print(f"Direction confirmed (α1 < 0): {alpha1 < 0}")

# ── 5. H2 – Time pressure increases overconfidence ───────────────────────────
# OLS: calibration_gap_ij = δ0 + δ1·TimePressure_i + γ_j + η_ij

model_h2 = smf.ols(
    "calibration_gap ~ time_pressure + C(question_id)",
    data=df
).fit(
    cov_type="cluster",
    cov_kwds={"groups": df["participant_id"]}
)

print("\n── H2: Effect of Time Pressure on Calibration Gap ──")
print(model_h2.summary().tables[1])
delta1 = model_h2.params["time_pressure"]
p_h2   = model_h2.pvalues["time_pressure"]
print(f"\nδ1 = {delta1:.4f}  (p = {p_h2:.4f})")
print(f"Direction confirmed (δ1 > 0): {delta1 > 0}")

# ── 6. Supplementary: two-sample t-tests (participant-level means) ────────────

pid_means = df.groupby(["participant_id", "time_pressure"])[
    ["accuracy", "confidence", "calibration_gap"]
].mean().reset_index()

ctrl = pid_means[pid_means["time_pressure"] == 0]
trt  = pid_means[pid_means["time_pressure"] == 1]

print("\n── Two-sample t-tests (participant-level means) ──")
for var in ["accuracy", "confidence", "calibration_gap"]:
    t, p = stats.ttest_ind(ctrl[var], trt[var], equal_var=False)
    print(f"  {var:20s}: t = {t:+.3f}, p = {p:.4f}")

# ── 7. Plots ─────────────────────────────────────────────────────────────────

COLORS = {"Control": "#4C72B0", "Treatment": "#DD8452"}
labels = {0: "Control", 1: "Treatment"}

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle("Time Pressure: Accuracy & Confidence Calibration", fontsize=13, fontweight="bold")

# 7a. Mean accuracy by condition
means_acc = pid_means.groupby("time_pressure")["accuracy"].mean()
axes[0].bar(
    [labels[k] for k in means_acc.index],
    means_acc.values,
    color=[COLORS[labels[k]] for k in means_acc.index],
    edgecolor="black", width=0.5
)
axes[0].set_ylim(0, 1)
axes[0].set_title("Mean Accuracy")
axes[0].set_ylabel("Proportion correct (±5 pp)")
for i, (k, v) in enumerate(means_acc.items()):
    axes[0].text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=10)

# 7b. Calibration gap distribution
for k, grp in pid_means.groupby("time_pressure"):
    axes[1].hist(grp["calibration_gap"], alpha=0.6, label=labels[k],
                 color=COLORS[labels[k]], bins=12, edgecolor="white")
axes[1].axvline(0, color="black", linestyle="--", linewidth=1)
axes[1].set_title("Calibration Gap Distribution")
axes[1].set_xlabel("Confidence − Accuracy (pp)\n> 0 = overconfident")
axes[1].set_ylabel("Frequency")
axes[1].legend()

# 7c. Mean confidence by condition
means_conf = pid_means.groupby("time_pressure")["confidence"].mean()
axes[2].bar(
    [labels[k] for k in means_conf.index],
    means_conf.values,
    color=[COLORS[labels[k]] for k in means_conf.index],
    edgecolor="black", width=0.5
)
axes[2].set_ylim(0, 100)
axes[2].set_title("Mean Confidence")
axes[2].set_ylabel("Confidence (1–100 scale)")
for i, (k, v) in enumerate(means_conf.items()):
    axes[2].text(i, v + 1.5, f"{v:.1f}", ha="center", fontsize=10)

plt.tight_layout()
plt.savefig("results_figure.png", dpi=150, bbox_inches="tight")
print("\nFigure saved to results_figure.png")

# ── 8. Results summary table ─────────────────────────────────────────────────

summary = pd.DataFrame({
    "Hypothesis": ["H1: accuracy", "H2: calibration gap"],
    "Coefficient": [alpha1, delta1],
    "p-value": [p_h1, p_h2],
    "Significant (p<0.05)": [p_h1 < 0.05, p_h2 < 0.05],
    "Direction confirmed": [alpha1 < 0, delta1 > 0],
})
print("\n── Hypothesis Summary ──")
print(summary.to_string(index=False))

print("\nDone. All pre-registered analyses complete.")