"""
Time Pressure, Accuracy, and Confidence Calibration
Analysis Script — Updated
-----------------------------------------------------
Main analysis: participant-level OLS using continuous dependent variables
  - H1: AbsoluteError_i = α0 + α1·TimePressure_i + u_i
  - H2: CalibrationGap_i = δ0 + δ1·TimePressure_i + η_i
        where CalibrationGap_i = mean(Confidence_i) − (100 − mean(AbsoluteError_i))

Robustness checks: question-level OLS with question fixed effects
  and standard errors clustered at the participant level
  - H1: AbsoluteError_ij = α0 + α1·TimePressure_i + γ_j + u_ij
  - H2: CalibrationGap_ij = δ0 + δ1·TimePressure_i + γ_j + η_ij
        where CalibrationGap_ij = Confidence_ij − (100 − AbsoluteError_ij)

Questions and correct answers:
  Q1 – Left-handed adults globally:          10%
  Q2 – Usable freshwater on Earth:            1%
  Q3 – Humans as share of mammal species:    36%
  Q4 – Share of emails that are spam:        45.6%
  Q5 – Smartphone checked within 5 min:      61%
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
    1: 10.0,
    2:  1.0,
    3: 36.0,
    4: 45.6,
    5: 61.0,
}

# ── 1. Load data ─────────────────────────────────────────────────────────────

DATA_PATH = "data1.csv"
_raw = pd.read_csv(DATA_PATH, skiprows=[1, 2])

def _reshape_qualtrics(raw):
    col_map = {
        (1, 1): ("Q1_15_1",  "Q2_C_15_1"),
        (2, 1): ("Q2_15_1",  "Q2_C_15_1.1"),
        (3, 1): ("Q3_15_1",  "Q3_C_15_1"),
        (4, 1): ("Q4_15_1",  "Q4_C_15_1"),
        (5, 1): ("Q5-15_1",  "Q5_C_15_1"),
        (1, 0): ("Q1_50_1",  "Q1_C_50_1"),
        (2, 0): ("Q2_50_1",  "Q2_C_50_1"),
        (3, 0): ("Q3_50_1",  "Q3_C_50_1"),
        (4, 0): ("Q4_50_1",  "Q4_C_50_1"),
        (5, 0): ("Q5_50_1",  "Q5_C_50_1"),
    }
    rows = []
    for _, p in raw.iterrows():
        pid   = p["ResponseId"]
        in_15 = pd.notna(p.get("Q1_15_1"))
        in_50 = pd.notna(p.get("Q1_50_1"))
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

if df["condition"].dtype == object:
    df["time_pressure"] = (df["condition"].str.lower() == "treatment").astype(int)
else:
    df["time_pressure"] = df["condition"].astype(int)

df["true_value"] = df["question_id"].map(TRUE_VALUES)

# ── 2. Derived variables ─────────────────────────────────────────────────────

# Continuous: absolute error
df["abs_error"] = np.abs(df["response"] - df["true_value"])

# Continuous calibration gap: Confidence − (100 − AbsoluteError)
# Positive → overconfident; Negative → underconfident
df["calibration_gap"] = df["confidence"] - (100 - df["abs_error"])

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
    df.groupby("time_pressure")[["abs_error", "confidence", "calibration_gap"]]
    .agg(["mean", "std"])
    .round(3)
)
desc.index = ["Control", "Treatment"]
print("\n── Descriptive Statistics ──────────────────────")
print(desc.to_string())

q_labels = {
    1: "Q1 Left-handed (10%)",
    2: "Q2 Freshwater (1%)",
    3: "Q3 Mammals (36%)",
    4: "Q4 Spam (45.6%)",
    5: "Q5 Phone (61%)",
}
q_err = df.groupby(["question_id", "time_pressure"])["abs_error"].mean().unstack()
q_err.columns = ["Control", "Treatment"]
q_err.index = [q_labels[int(str(i))] for i in q_err.index]
print("\n── Mean Absolute Error by Question ─────────────")
print(q_err.round(3).to_string())

# ── 5. MAIN ANALYSIS — Participant-level OLS ──────────────────────────────────

pid_df = df.groupby(["participant_id", "time_pressure"]).agg(
    abs_error       = ("abs_error",       "mean"),
    confidence      = ("confidence",      "mean"),
    calibration_gap = ("calibration_gap", "mean"),
).reset_index()

# H1: AbsoluteError_i = α0 + α1·TimePressure_i + u_i
model_h1 = smf.ols("abs_error ~ time_pressure", data=pid_df).fit()

alpha1 = model_h1.params["time_pressure"]
se_h1  = model_h1.bse["time_pressure"]
ci_h1  = model_h1.conf_int().loc["time_pressure"]
p_h1   = model_h1.pvalues["time_pressure"]

print("\n── H1: Effect of Time Pressure on Absolute Error (participant level) ──")
print(model_h1.summary().tables[1])
print(f"\nα1 = {alpha1:+.4f}  SE = {se_h1:.4f}  "
      f"95% CI [{ci_h1[0]:+.4f}, {ci_h1[1]:+.4f}]  p = {p_h1:.4f}")
print(f"Predicted direction (α1 > 0) confirmed: {alpha1 > 0}")

# H2: CalibrationGap_i = δ0 + δ1·TimePressure_i + η_i
pid_h2 = pid_df.dropna(subset=["calibration_gap"])
model_h2 = smf.ols("calibration_gap ~ time_pressure", data=pid_h2).fit()

delta1 = model_h2.params["time_pressure"]
se_h2  = model_h2.bse["time_pressure"]
ci_h2  = model_h2.conf_int().loc["time_pressure"]
p_h2   = model_h2.pvalues["time_pressure"]

print("\n── H2: Effect of Time Pressure on Calibration Gap (participant level) ──")
print(model_h2.summary().tables[1])
print(f"\nδ1 = {delta1:+.4f}  SE = {se_h2:.4f}  "
      f"95% CI [{ci_h2[0]:+.4f}, {ci_h2[1]:+.4f}]  p = {p_h2:.4f}")
print(f"Predicted direction (δ1 > 0) confirmed: {delta1 > 0}")

# ── 6. ROBUSTNESS CHECKS — Question-level OLS with fixed effects ──────────────

print("\n── Robustness Check: Question-level OLS with fixed effects ─────────────")

# H1 question-level
df_h1q = df.dropna(subset=["abs_error"])
model_h1q = smf.ols(
    "abs_error ~ time_pressure + C(question_id)",
    data=df_h1q
).fit(
    cov_type="cluster",
    cov_kwds={"groups": df_h1q["participant_id"]}
)

alpha1q = model_h1q.params["time_pressure"]
se_h1q  = model_h1q.bse["time_pressure"]
ci_h1q  = model_h1q.conf_int().loc["time_pressure"]
p_h1q   = model_h1q.pvalues["time_pressure"]

print("\nH1 (question-level):")
print(model_h1q.summary().tables[1])
print(f"\nα1 = {alpha1q:+.4f}  SE = {se_h1q:.4f}  "
      f"95% CI [{ci_h1q[0]:+.4f}, {ci_h1q[1]:+.4f}]  p = {p_h1q:.4f}")

# H2 question-level
df_h2q = df.dropna(subset=["calibration_gap"])
model_h2q = smf.ols(
    "calibration_gap ~ time_pressure + C(question_id)",
    data=df_h2q
).fit(
    cov_type="cluster",
    cov_kwds={"groups": df_h2q["participant_id"]}
)

delta1q = model_h2q.params["time_pressure"]
se_h2q  = model_h2q.bse["time_pressure"]
ci_h2q  = model_h2q.conf_int().loc["time_pressure"]
p_h2q   = model_h2q.pvalues["time_pressure"]

print("\nH2 (question-level):")
print(model_h2q.summary().tables[1])
print(f"\nδ1 = {delta1q:+.4f}  SE = {se_h2q:.4f}  "
      f"95% CI [{ci_h2q[0]:+.4f}, {ci_h2q[1]:+.4f}]  p = {p_h2q:.4f}")

# ── 7. Welch t-tests (participant-level) ─────────────────────────────────────

ctrl = pid_df[pid_df["time_pressure"] == 0]
trt  = pid_df[pid_df["time_pressure"] == 1]

print("\n── Welch t-tests (participant-level means) ─────")
for var in ["abs_error", "confidence", "calibration_gap"]:
    c_vals = ctrl[var].dropna()
    t_vals = trt[var].dropna()
    t_stat, p_val = stats.ttest_ind(c_vals, t_vals, equal_var=False)
    pooled_sd = np.sqrt((c_vals.std()**2 + t_vals.std()**2) / 2)
    d = (t_vals.mean() - c_vals.mean()) / pooled_sd if pooled_sd > 0 else np.nan
    print(f"  {var:20s}: t = {t_stat:+.3f},  p = {p_val:.4f},  Cohen's d = {d:+.3f}")

# ── 8. Plots ─────────────────────────────────────────────────────────────────

COLORS      = {"Control": "#4C72B0", "Treatment": "#DD8452"}
cond_labels = {0: "Control", 1: "Treatment"}

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(
    "Time Pressure: Absolute Error & Confidence Calibration",
    fontsize=13, fontweight="bold"
)

# 8a & 8c — bar plots
for col, ax, title, ylabel, ylim in [
    ("abs_error",       axes[0], "Mean Absolute Error",  "Absolute Error (pp)",  (0, 50)),
    ("confidence",      axes[2], "Mean Confidence",      "Confidence (%)",        (0, 100)),
]:
    grp   = pid_df.groupby("time_pressure")[col]
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
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel)
    for i, (k, v) in enumerate(means.items()):
        offset = ylim[1] * 0.03
        ax.text(i, v + sems.iloc[i] + offset, f"{v:.2f}", ha="center", fontsize=10)

# 8b — calibration gap box + strip
cond_order = [0, 1]
box_data = [
    pid_df[pid_df["time_pressure"] == k]["calibration_gap"].dropna().values
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
axes[1].set_title("Calibration Gap", fontsize=12, fontweight="bold")
axes[1].set_ylabel("Confidence − (100 − AbsError)\n> 0 = overconfident")

plt.tight_layout()
plt.savefig("results_figure.png", dpi=150, bbox_inches="tight")
print("\nFigure saved → results_figure.png")

# ── 9. Summary table ─────────────────────────────────────────────────────────

summary = pd.DataFrame({
    "Hypothesis":  [
        "H1 main (participant-level)",
        "H2 main (participant-level)",
        "H1 robustness (question-level + FE)",
        "H2 robustness (question-level + FE)",
    ],
    "Coefficient": [f"{alpha1:+.4f}", f"{delta1:+.4f}", f"{alpha1q:+.4f}", f"{delta1q:+.4f}"],
    "SE":          [f"{se_h1:.4f}",   f"{se_h2:.4f}",   f"{se_h1q:.4f}",  f"{se_h2q:.4f}"],
    "95% CI":      [
        f"[{ci_h1[0]:+.4f}, {ci_h1[1]:+.4f}]",
        f"[{ci_h2[0]:+.4f}, {ci_h2[1]:+.4f}]",
        f"[{ci_h1q[0]:+.4f}, {ci_h1q[1]:+.4f}]",
        f"[{ci_h2q[0]:+.4f}, {ci_h2q[1]:+.4f}]",
    ],
    "p-value":     [f"{p_h1:.4f}", f"{p_h2:.4f}", f"{p_h1q:.4f}", f"{p_h2q:.4f}"],
    "p < 0.05":    [p_h1 < 0.05, p_h2 < 0.05, p_h1q < 0.05, p_h2q < 0.05],
    "Direction ✓": [alpha1 > 0,  delta1 > 0,  alpha1q > 0,  delta1q > 0],
})
print("\n── Results Summary ─────────────────────────────")
print(summary.to_string(index=False))
print("\nDone. All analyses complete.")