"""
Time Pressure, Accuracy, and Confidence Calibration
Pre-registered Analysis Script
-----------------------------------------------------
Questions and correct answers:
  Q1 – Left-handed adults globally:          10%
  Q2 – Usable freshwater on Earth:            1%
  Q3 – Humans as share of mammal species:    36%
  Q4 – Share of emails that are spam:        45.6%
  Q5 – Smartphone checked within 5 min:      61%

Expected CSV columns (one row per participant × question):
  participant_id : unique participant identifier
  condition      : 'treatment' or 'control'  (or 1 / 0)
  question_id    : integer 1–5
  response       : participant's numeric estimate (0–100)
  confidence     : self-reported confidence (1–100 scale, in %)
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ── 0. Ground-truth answers ──────────────────────────────────────────────────

TRUE_VALUES = {
    1: 10.0,   # % left-handed adults
    2:  1.0,   # % usable freshwater
    3: 36.0,   # % mammal species that are human
    4: 45.6,   # % emails that are spam
    5: 61.0,   # % checking phone within 5 min of waking
}

ACCURACY_THRESHOLD = 5   # ±5 percentage points

# ── 1. Load data ─────────────────────────────────────────────────────────────

DATA_PATH = "data1.csv"   # ← update to your actual file path
_raw = pd.read_csv(DATA_PATH, skiprows=[1, 2])  # skip Qualtrics description + ImportId rows

def _reshape_qualtrics(raw):
    """Convert wide Qualtrics export to long format expected by analysis."""
    # Column mapping: (question_id, condition) → (response_col, confidence_col)
    col_map = {
        (1, 1): ("Q1_15_1",   "Q2_C_15_1"),      # Q1 15s: conf col misnamed in survey
        (2, 1): ("Q2_15_1",   "Q2_C_15_1.1"),    # Q2 15s
        (3, 1): ("Q3_15_1",   "Q3_C_15_1"),
        (4, 1): ("Q4_15_1",   "Q4_C_15_1"),
        (5, 1): ("Q5-15_1",   "Q5_C_15_1"),
        (1, 0): ("Q1_50_1",   "Q1_C_50_1"),
        (2, 0): ("Q2_50_1",   "Q2_C_50_1"),
        (3, 0): ("Q3_50_1",   "Q3_C_50_1"),
        (4, 0): ("Q4_50_1",   "Q4_C_50_1"),
        (5, 0): ("Q5_50_1",   "Q5_C_50_1"),
    }
    rows = []
    for _, p in raw.iterrows():
        pid = p["ResponseId"]
        in_15 = pd.notna(p.get("Q1_15_1"))
        in_50 = pd.notna(p.get("Q1_50_1"))
        # Prefer 15s if both non-null (edge case); skip if neither
        if in_15:
            cond = 1
        elif in_50:
            cond = 0
        else:
            continue
        for qid in range(1, 6):
            r_col, c_col = col_map[(qid, cond)]
            resp = p.get(r_col)
            conf = p.get(c_col)
            if pd.isna(resp):
                continue
            rows.append({
                "participant_id": pid,
                "condition":      cond,
                "question_id":    qid,
                "response":       float(resp),
                "confidence":     float(conf) if pd.notna(conf) else np.nan,
            })
    return pd.DataFrame(rows)

if "condition" in _raw.columns:
    df = _raw.copy()
else:
    df = _reshape_qualtrics(_raw)

# Normalise condition column to 0 / 1
if df["condition"].dtype == object:
    df["time_pressure"] = (df["condition"].str.lower() == "treatment").astype(int)
else:
    df["time_pressure"] = df["condition"].astype(int)

# Map true values onto each row
df["true_value"] = df["question_id"].map(TRUE_VALUES)

# ── 2. Derived variables ─────────────────────────────────────────────────────

# Accuracy: 1 if |response - true_value| ≤ 5 pp, else 0
df["accuracy"] = (
    np.abs(df["response"] - df["true_value"]) <= ACCURACY_THRESHOLD
).astype(int)

# Calibration gap (confidence already on 0–100 scale)
# CalibrationGap = Confidence − (100 × Accuracy)
# Positive → overconfident; negative → underconfident
df["calibration_gap"] = df["confidence"] - (100 * df["accuracy"])

df["question_id"] = df["question_id"].astype("category")

# ── 3. Exclusion rule ────────────────────────────────────────────────────────

q_per_participant = df.groupby("participant_id")["question_id"].count()
keep = q_per_participant[q_per_participant >= 4].index
n_before = df["participant_id"].nunique()
df = df[df["participant_id"].isin(keep)].copy()
n_after = df["participant_id"].nunique()

print("── Sample ──────────────────────────────────────")
print(f"Participants before exclusion : {n_before}")
print(f"Participants after exclusion  : {n_after}  (kept ≥4 questions)")
print(f"  Control   : {df[df['time_pressure']==0]['participant_id'].nunique()}")
print(f"  Treatment : {df[df['time_pressure']==1]['participant_id'].nunique()}")

# ── 4. Descriptive statistics ────────────────────────────────────────────────

desc = (
    df.groupby("time_pressure")[["accuracy", "confidence", "calibration_gap"]]
    .agg(["mean", "std"])
    .round(3)
)
desc.index = ["Control", "Treatment"]
print("\n── Descriptive Statistics ──────────────────────")
print(desc.to_string())

# Per-question accuracy rates
q_acc = df.groupby(["question_id", "time_pressure"])["accuracy"].mean().unstack()
q_acc.columns = ["Control", "Treatment"]
print("\n── Accuracy by Question ────────────────────────")
q_labels = {
    1: "Q1 Left-handed (10%)",
    2: "Q2 Freshwater (1%)",
    3: "Q3 Mammals (36%)",
    4: "Q4 Spam (45.6%)",
    5: "Q5 Phone (61%)",
}
q_acc.index = [q_labels[int(str(i))] for i in q_acc.index]
print(q_acc.round(3).to_string())

# ── 5. Create participant-level dataset ─────────────────────────

pid_means = df.groupby(["participant_id","time_pressure"])[
    ["accuracy","confidence","calibration_gap"]
].mean().reset_index()

# ── 6. H1 – Time pressure reduces accuracy (participant-level) ─

model_h1 = smf.ols(
    "accuracy ~ time_pressure",
    data=pid_means
).fit()

alpha1 = model_h1.params["time_pressure"]
se_h1  = model_h1.bse["time_pressure"]
ci_h1  = model_h1.conf_int().loc["time_pressure"]
p_h1   = model_h1.pvalues["time_pressure"]

print("\n── H1: Effect of Time Pressure on Accuracy ─────")
print(model_h1.summary())

# ── 7. H2 – Time pressure increases calibration gap ────────────

model_h2 = smf.ols(
    "calibration_gap ~ time_pressure",
    data=pid_means
).fit()

delta1 = model_h2.params["time_pressure"]
se_h2  = model_h2.bse["time_pressure"]
ci_h2  = model_h2.conf_int().loc["time_pressure"]
p_h2   = model_h2.pvalues["time_pressure"]

print("\n── H2: Effect of Time Pressure on Calibration Gap ──")
print(model_h2.summary())

# ── 8. Plots ─────────────────────────────────────────────────────────────────

COLORS = {"Control": "#6F44A3", "Treatment": "#DD8452"}
cond_labels = {0: "Control", 1: "Treatment"}

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(
    "Time Pressure: Accuracy & Confidence Calibration\n"
    f"(accuracy threshold = ±{ACCURACY_THRESHOLD} pp)",
    fontsize=13, fontweight="bold"
)

# 8a. Mean accuracy by condition with error bars
for col, ax, title, ylabel, ylim in [
    ("accuracy",   axes[0], "Mean Accuracy",    f"Proportion correct (±{ACCURACY_THRESHOLD} pp)", (0, 1)),
    ("confidence", axes[2], "Mean Confidence",  "Confidence (%)",                                  (0, 100)),
]:
    grp   = pid_means.groupby("time_pressure")[col]
    means = grp.mean()
    sems  = grp.sem()
    ax.bar(
        [cond_labels[k] for k in means.index],
        means.values,
        yerr=sems.values,
        color=[COLORS[cond_labels[k]] for k in means.index],
        edgecolor="black", width=0.5, capsize=5
    )
    ax.set_ylim(*ylim)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    for i, (k, v) in enumerate(means.items()):
        offset = ylim[1] * 0.03
        ax.text(i, v + sems.iloc[i] + offset, f"{v:.2f}", ha="center", fontsize=10)

# 8b. Calibration gap — box + strip plot
cond_order = [0, 1]
box_data = [
    pid_means[pid_means["time_pressure"] == k]["calibration_gap"].values
    for k in cond_order
]
bp = axes[1].boxplot(
    box_data,
    labels=[cond_labels[k] for k in cond_order],
    patch_artist=True,
    medianprops=dict(color="black", linewidth=2)
)
for patch, k in zip(bp["boxes"], cond_order):
    patch.set_facecolor(COLORS[cond_labels[k]])
    patch.set_alpha(0.7)

rng = np.random.default_rng(42)
for i, k in enumerate(cond_order):
    jitter = rng.uniform(-0.08, 0.08, size=len(box_data[i]))
    axes[1].scatter(
        np.full(len(box_data[i]), i + 1) + jitter,
        box_data[i],
        alpha=0.5, s=20, color=COLORS[cond_labels[k]], zorder=3
    )

axes[1].axhline(0, color="black", linestyle="--", linewidth=1)
axes[1].set_title("Calibration Gap")
axes[1].set_ylabel("Confidence − Accuracy (pp)\n> 0 = overconfident")

plt.tight_layout()
plt.savefig("results_figure_participant.png", dpi=150, bbox_inches="tight")
print("\nFigure saved → results_figure_participant.png")

# ── 9. Hypothesis summary ────────────────────────────────────────────────────

summary = pd.DataFrame({
    "Hypothesis":  ["H1: time pressure → ↓ accuracy",
                    "H2: time pressure → ↑ calibration gap"],
    "Coefficient": [f"{alpha1:+.4f}", f"{delta1:+.4f}"],
    "SE":          [f"{se_h1:.4f}",   f"{se_h2:.4f}"],
    "95% CI":      [f"[{ci_h1[0]:+.4f}, {ci_h1[1]:+.4f}]",
                    f"[{ci_h2[0]:+.4f}, {ci_h2[1]:+.4f}]"],
    "p-value":     [f"{p_h1:.4f}",    f"{p_h2:.4f}"],
    "p < 0.05":    [p_h1 < 0.05,      p_h2 < 0.05],
    "Direction ✓": [alpha1 < 0,        delta1 > 0],
})
print("\n── Hypothesis Summary ──────────────────────────")
print(summary.to_string(index=False))
print("\nDone. All pre-registered analyses complete.")