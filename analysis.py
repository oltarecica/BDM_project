"""
Time Pressure, Accuracy, and Confidence Calibration
Analysis Script
-----------------------------------------------------
Structure:
  MAIN ANALYSIS   — participant-level OLS, binary DV
  EXTENSION       — participant-level OLS, continuous DV
  ROBUSTNESS      — question-level OLS with question FE + clustered SEs
                    for both binary and continuous specifications

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

ACCURACY_THRESHOLD = 5   # ±5 percentage points

# ── 1. Load & reshape data ───────────────────────────────────────────────────

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

# Binary accuracy: 1 if |response − true_value| ≤ 5 pp
df["accuracy"] = (
    np.abs(df["response"] - df["true_value"]) <= ACCURACY_THRESHOLD
).astype(float)
df.loc[df["response"].isna(), "accuracy"] = np.nan

# Binary calibration gap: Confidence − (100 × Accuracy)
df["calib_gap_binary"] = df["confidence"] - (100 * df["accuracy"])

# Continuous: absolute error
df["abs_error"] = np.abs(df["response"] - df["true_value"])

# Continuous calibration gap: Confidence − (100 − AbsoluteError)
df["calib_gap_cont"] = df["confidence"] - (100 - df["abs_error"])

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
    df.groupby("time_pressure")[["accuracy", "abs_error", "confidence",
                                  "calib_gap_binary", "calib_gap_cont"]]
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
q_acc = df.groupby(["question_id", "time_pressure"])["accuracy"].mean().unstack()
q_acc.columns = ["Control", "Treatment"]
q_acc.index = [q_labels[int(str(i))] for i in q_acc.index]
print("\n── Accuracy Rate by Question ────────────────────")
print(q_acc.round(3).to_string())

# ── 5. MAIN ANALYSIS — Participant-level, binary ──────────────────────────────

pid_df = df.groupby(["participant_id", "time_pressure"]).agg(
    accuracy        = ("accuracy",        "mean"),
    abs_error       = ("abs_error",       "mean"),
    confidence      = ("confidence",      "mean"),
    calib_gap_binary= ("calib_gap_binary","mean"),
    calib_gap_cont  = ("calib_gap_cont",  "mean"),
).reset_index()

print("\n══════════════════════════════════════════════════")
print("  MAIN ANALYSIS — Participant-level, binary DV")
print("══════════════════════════════════════════════════")

# H1 binary: Accuracy_i = α0 + α1·TimePressure_i + u_i
model_h1 = smf.ols("accuracy ~ time_pressure", data=pid_df).fit()
alpha1 = model_h1.params["time_pressure"]
se_h1  = model_h1.bse["time_pressure"]
ci_h1  = model_h1.conf_int().loc["time_pressure"]
p_h1   = model_h1.pvalues["time_pressure"]

print("\nH1: Accuracy_i = α0 + α1·TimePressure_i + u_i")
print(model_h1.summary().tables[1])
print(f"α1 = {alpha1:+.4f}  SE = {se_h1:.4f}  95% CI [{ci_h1[0]:+.4f}, {ci_h1[1]:+.4f}]  p = {p_h1:.4f}")
print(f"Predicted direction (α1 < 0) confirmed: {alpha1 < 0}")

# H2 binary: CalibrationGap_i = δ0 + δ1·TimePressure_i + η_i
pid_h2 = pid_df.dropna(subset=["calib_gap_binary"])
model_h2 = smf.ols("calib_gap_binary ~ time_pressure", data=pid_h2).fit()
delta1 = model_h2.params["time_pressure"]
se_h2  = model_h2.bse["time_pressure"]
ci_h2  = model_h2.conf_int().loc["time_pressure"]
p_h2   = model_h2.pvalues["time_pressure"]

print("\nH2: CalibrationGap_i = δ0 + δ1·TimePressure_i + η_i")
print("    where CalibrationGap_i = mean(Confidence_i) − 100 × mean(Accuracy_i)")
print(model_h2.summary().tables[1])
print(f"δ1 = {delta1:+.4f}  SE = {se_h2:.4f}  95% CI [{ci_h2[0]:+.4f}, {ci_h2[1]:+.4f}]  p = {p_h2:.4f}")
print(f"Predicted direction (δ1 > 0) confirmed: {delta1 > 0}")

# ── 6. EXTENSION — Participant-level, continuous DV ───────────────────────────

print("\n══════════════════════════════════════════════════")
print("  EXTENSION — Participant-level, continuous DV")
print("══════════════════════════════════════════════════")

# H1 continuous: AbsoluteError_i = α0 + α1·TimePressure_i + u_i
model_h1c = smf.ols("abs_error ~ time_pressure", data=pid_df).fit()
alpha1c = model_h1c.params["time_pressure"]
se_h1c  = model_h1c.bse["time_pressure"]
ci_h1c  = model_h1c.conf_int().loc["time_pressure"]
p_h1c   = model_h1c.pvalues["time_pressure"]

print("\nH1: AbsoluteError_i = α0 + α1·TimePressure_i + u_i")
print(model_h1c.summary().tables[1])
print(f"α1 = {alpha1c:+.4f}  SE = {se_h1c:.4f}  95% CI [{ci_h1c[0]:+.4f}, {ci_h1c[1]:+.4f}]  p = {p_h1c:.4f}")
print(f"Predicted direction (α1 > 0) confirmed: {alpha1c > 0}")

# H2 continuous: CalibrationGap_i = δ0 + δ1·TimePressure_i + η_i
pid_h2c = pid_df.dropna(subset=["calib_gap_cont"])
model_h2c = smf.ols("calib_gap_cont ~ time_pressure", data=pid_h2c).fit()
delta1c = model_h2c.params["time_pressure"]
se_h2c  = model_h2c.bse["time_pressure"]
ci_h2c  = model_h2c.conf_int().loc["time_pressure"]
p_h2c   = model_h2c.pvalues["time_pressure"]

print("\nH2: CalibrationGap_i = δ0 + δ1·TimePressure_i + η_i")
print("    where CalibrationGap_i = mean(Confidence_i) − (100 − mean(AbsoluteError_i))")
print(model_h2c.summary().tables[1])
print(f"δ1 = {delta1c:+.4f}  SE = {se_h2c:.4f}  95% CI [{ci_h2c[0]:+.4f}, {ci_h2c[1]:+.4f}]  p = {p_h2c:.4f}")
print(f"Predicted direction (δ1 > 0) confirmed: {delta1c > 0}")

# ── 7. ROBUSTNESS — Question-level with fixed effects + clustered SEs ─────────

print("\n══════════════════════════════════════════════════")
print("  ROBUSTNESS — Question-level, FE + clustered SEs")
print("══════════════════════════════════════════════════")

def run_fe_model(formula, data, cluster_var):
    return smf.ols(formula, data=data).fit(
        cov_type="cluster",
        cov_kwds={"groups": data[cluster_var]}
    )

# H1 binary
df_r = df.dropna(subset=["accuracy"])
m_r_h1 = run_fe_model("accuracy ~ time_pressure + C(question_id)", df_r, "participant_id")
a1r = m_r_h1.params["time_pressure"]; s1r = m_r_h1.bse["time_pressure"]
c1r = m_r_h1.conf_int().loc["time_pressure"]; p1r = m_r_h1.pvalues["time_pressure"]
print("\nH1 binary (question-level + FE):")
print(m_r_h1.summary().tables[1])
print(f"α1 = {a1r:+.4f}  SE = {s1r:.4f}  95% CI [{c1r[0]:+.4f}, {c1r[1]:+.4f}]  p = {p1r:.4f}")

# H2 binary
df_r2 = df.dropna(subset=["calib_gap_binary"])
m_r_h2 = run_fe_model("calib_gap_binary ~ time_pressure + C(question_id)", df_r2, "participant_id")
d1r = m_r_h2.params["time_pressure"]; s2r = m_r_h2.bse["time_pressure"]
c2r = m_r_h2.conf_int().loc["time_pressure"]; p2r = m_r_h2.pvalues["time_pressure"]
print("\nH2 binary (question-level + FE):")
print(m_r_h2.summary().tables[1])
print(f"δ1 = {d1r:+.4f}  SE = {s2r:.4f}  95% CI [{c2r[0]:+.4f}, {c2r[1]:+.4f}]  p = {p2r:.4f}")

# H1 continuous
df_r3 = df.dropna(subset=["abs_error"])
m_r_h1c = run_fe_model("abs_error ~ time_pressure + C(question_id)", df_r3, "participant_id")
a1rc = m_r_h1c.params["time_pressure"]; s1rc = m_r_h1c.bse["time_pressure"]
c1rc = m_r_h1c.conf_int().loc["time_pressure"]; p1rc = m_r_h1c.pvalues["time_pressure"]
print("\nH1 continuous (question-level + FE):")
print(m_r_h1c.summary().tables[1])
print(f"α1 = {a1rc:+.4f}  SE = {s1rc:.4f}  95% CI [{c1rc[0]:+.4f}, {c1rc[1]:+.4f}]  p = {p1rc:.4f}")

# H2 continuous
df_r4 = df.dropna(subset=["calib_gap_cont"])
m_r_h2c = run_fe_model("calib_gap_cont ~ time_pressure + C(question_id)", df_r4, "participant_id")
d1rc = m_r_h2c.params["time_pressure"]; s2rc = m_r_h2c.bse["time_pressure"]
c2rc = m_r_h2c.conf_int().loc["time_pressure"]; p2rc = m_r_h2c.pvalues["time_pressure"]
print("\nH2 continuous (question-level + FE):")
print(m_r_h2c.summary().tables[1])
print(f"δ1 = {d1rc:+.4f}  SE = {s2rc:.4f}  95% CI [{c2rc[0]:+.4f}, {c2rc[1]:+.4f}]  p = {p2rc:.4f}")

# ── 8. Welch t-tests ─────────────────────────────────────────────────────────

ctrl = pid_df[pid_df["time_pressure"] == 0]
trt  = pid_df[pid_df["time_pressure"] == 1]

print("\n── Welch t-tests (participant-level means) ─────")
for var in ["accuracy", "abs_error", "confidence", "calib_gap_binary", "calib_gap_cont"]:
    c_vals = ctrl[var].dropna()
    t_vals = trt[var].dropna()
    t_stat, p_val = stats.ttest_ind(c_vals, t_vals, equal_var=False)
    pooled_sd = np.sqrt((c_vals.std()**2 + t_vals.std()**2) / 2)
    d = (t_vals.mean() - c_vals.mean()) / pooled_sd if pooled_sd > 0 else np.nan
    print(f"  {var:22s}: t = {t_stat:+.3f},  p = {p_val:.4f},  Cohen's d = {d:+.3f}")

# ── 9. Plots ─────────────────────────────────────────────────────────────────

COLORS      = {"Control": "#4C72B0", "Treatment": "#DD8452"}
cond_labels = {0: "Control", 1: "Treatment"}

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle("Time Pressure: Accuracy & Confidence Calibration", fontsize=14, fontweight="bold")

# Row 1: Binary
for col, ax, title, ylabel, ylim in [
    ("accuracy",        axes[0][0], "Mean Accuracy (Binary)",       f"Proportion correct (±{ACCURACY_THRESHOLD} pp)", (0, 1)),
    ("confidence",      axes[0][2], "Mean Confidence",               "Confidence (%)",                                 (0, 100)),
]:
    grp   = pid_df.groupby("time_pressure")[col]
    means = grp.mean()
    sems  = grp.sem()
    ax.bar([cond_labels[k] for k in means.index], means.values,
           yerr=sems.values, color=[COLORS[cond_labels[k]] for k in means.index],
           edgecolor="black", width=0.5, capsize=5)
    ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylabel(ylabel)
    for i, (k, v) in enumerate(means.items()):
        ax.text(i, v + sems.iloc[i] + ylim[1]*0.03, f"{v:.2f}", ha="center", fontsize=10)

# Calibration gap binary — box + strip
box_data = [pid_df[pid_df["time_pressure"]==k]["calib_gap_binary"].dropna().values for k in [0,1]]
bp = axes[0][1].boxplot(box_data, labels=["Control","Treatment"], patch_artist=True,
                         medianprops=dict(color="black", linewidth=2))
for patch, k in zip(bp["boxes"], [0,1]):
    patch.set_facecolor(COLORS[cond_labels[k]]); patch.set_alpha(0.7)
rng = np.random.default_rng(42)
for i, k in enumerate([0,1]):
    jitter = rng.uniform(-0.08, 0.08, size=len(box_data[i]))
    axes[0][1].scatter(np.full(len(box_data[i]), i+1)+jitter, box_data[i],
                       alpha=0.5, s=20, color=COLORS[cond_labels[k]], zorder=3)
axes[0][1].axhline(0, color="black", linestyle="--", linewidth=1)
axes[0][1].set_title("Calibration Gap (Binary)", fontsize=11, fontweight="bold")
axes[0][1].set_ylabel("Confidence − 100×Accuracy\n> 0 = overconfident")

# Row 2: Continuous
for col, ax, title, ylabel, ylim in [
    ("abs_error",  axes[1][0], "Mean Absolute Error (Continuous)", "Absolute Error (pp)", (0, 50)),
    ("confidence", axes[1][2], "Mean Confidence",                   "Confidence (%)",      (0, 100)),
]:
    grp   = pid_df.groupby("time_pressure")[col]
    means = grp.mean()
    sems  = grp.sem()
    ax.bar([cond_labels[k] for k in means.index], means.values,
           yerr=sems.values, color=[COLORS[cond_labels[k]] for k in means.index],
           edgecolor="black", width=0.5, capsize=5)
    ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylabel(ylabel)
    for i, (k, v) in enumerate(means.items()):
        ax.text(i, v + sems.iloc[i] + ylim[1]*0.03, f"{v:.2f}", ha="center", fontsize=10)

# Calibration gap continuous — box + strip
box_data2 = [pid_df[pid_df["time_pressure"]==k]["calib_gap_cont"].dropna().values for k in [0,1]]
bp2 = axes[1][1].boxplot(box_data2, labels=["Control","Treatment"], patch_artist=True,
                          medianprops=dict(color="black", linewidth=2))
for patch, k in zip(bp2["boxes"], [0,1]):
    patch.set_facecolor(COLORS[cond_labels[k]]); patch.set_alpha(0.7)
for i, k in enumerate([0,1]):
    jitter = rng.uniform(-0.08, 0.08, size=len(box_data2[i]))
    axes[1][1].scatter(np.full(len(box_data2[i]), i+1)+jitter, box_data2[i],
                       alpha=0.5, s=20, color=COLORS[cond_labels[k]], zorder=3)
axes[1][1].axhline(0, color="black", linestyle="--", linewidth=1)
axes[1][1].set_title("Calibration Gap (Continuous)", fontsize=11, fontweight="bold")
axes[1][1].set_ylabel("Confidence − (100 − AbsError)\n> 0 = overconfident")

plt.tight_layout()
plt.savefig("results_figure.png", dpi=150, bbox_inches="tight")
print("\nFigure saved → results_figure.png")

# ── 10. Summary table ────────────────────────────────────────────────────────

summary = pd.DataFrame({
    "Specification": [
        "H1 main — binary (participant)",
        "H2 main — binary (participant)",
        "H1 extension — continuous (participant)",
        "H2 extension — continuous (participant)",
        "H1 robustness — binary (question + FE)",
        "H2 robustness — binary (question + FE)",
        "H1 robustness — continuous (question + FE)",
        "H2 robustness — continuous (question + FE)",
    ],
    "Coef":    [f"{x:+.4f}" for x in [alpha1, delta1, alpha1c, delta1c, a1r, d1r, a1rc, d1rc]],
    "SE":      [f"{x:.4f}"  for x in [se_h1,  se_h2,  se_h1c,  se_h2c,  s1r, s2r, s1rc, s2rc]],
    "95% CI":  [f"[{a:+.4f}, {b:+.4f}]" for a, b in [
                    ci_h1, ci_h2, ci_h1c, ci_h2c, c1r, c2r, c1rc, c2rc]],
    "p-value": [f"{x:.4f}" for x in [p_h1, p_h2, p_h1c, p_h2c, p1r, p2r, p1rc, p2rc]],
    "p<0.05":  [x < 0.05   for x in [p_h1, p_h2, p_h1c, p_h2c, p1r, p2r, p1rc, p2rc]],
    "Dir ✓":   [alpha1<0, delta1>0, alpha1c>0, delta1c>0, a1r<0, d1r>0, a1rc>0, d1rc>0],
})
print("\n── Results Summary ─────────────────────────────")
print(summary.to_string(index=False))
print("\nDone. All analyses complete.")