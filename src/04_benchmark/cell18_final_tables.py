# ===== CELL 18 — Final paper tables and figures from saved out-of-fold scores (needs only Cell 1; restores files from Drive) =====
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, roc_curve
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
def down(sub, local, extra=()):
    os.makedirs(local, exist_ok=True)
    r = subprocess.run(["rclone", "copy", f"{RCLONE_REMOTE}:{DRIVE_SUBDIR}/{sub}", local, f"--drive-root-folder-id={DRIVE_FOLDER_ID}",
                        "--transfers=4", "-q", *extra], capture_output=True, text=True)
    print(("downloaded " if r.returncode == 0 else "!! FAILED ") + f"{sub} -> {local}", r.stderr[:300] if r.returncode else "")
down("benchmark", OUT, ["--include", "unified_oof_scores.csv"])
down("roi", OUT, ["--include", "roi_oof_scores.csv", "--include", "roi_agematched.csv"])
A = pd.read_csv(f"{OUT}/roi_oof_scores.csv"); U = pd.read_csv(f"{OUT}/unified_oof_scores.csv")
M = A.merge(U[["subject_id"] + [c for c in U.columns if c not in A.columns]], on="subject_id"); y = M.cdr_binary.values.astype(int); N = len(M)
ENS = "Ensemble (rank-mean of ROI MTL+vent, CNN age-pretrained, Blocks)"      # pre-specified, parameter-free fusion; exploratory
M[ENS] = np.mean([rankdata(M[k]) for k in ["ROI MTL+vent (MRI-only)", "CNN age-pretrained", "Blocks MRI-only (PCA+LR)"]], 0) / N
ORDER = ["nWBV only", "LR age+sex+eTIV+nWBV", "LR all volumetric+age+sex", "Block-ridge BAG (raw)", "CNN BAG (raw)", "CNN scratch", "CNN age-pretrained",
         "Blocks MRI-only (PCA+LR)", "ROI all (MRI-only)", "ROI MTL+vent (MRI-only)", "ROI MTL+vent + age,sex,eTIV,nWBV", ENS]
ORDER = [k for k in ORDER if k in M.columns]; PROB = [k for k in ORDER if k not in ("nWBV only", "Block-ridge BAG (raw)", "CNN BAG (raw)", ENS)]
rng = np.random.default_rng(0); BI = [b for b in (rng.integers(0, N, N) for _ in range(2000)) if len(np.unique(y[b])) == 2]
def sens_at_spec(s, spec=0.8): fpr, tpr, _ = roc_curve(y, s); return float(tpr[fpr <= 1 - spec].max())

# ---- Table 1: main results ----
rows = []
for k in ORDER:
    s = M[k].values; v = [roc_auc_score(y[b], s[b]) for b in BI]
    rows.append(dict(model=k, auroc=roc_auc_score(y, s), ci_lo=np.percentile(v, 2.5), ci_hi=np.percentile(v, 97.5), avg_precision=average_precision_score(y, s),
                     sens_at_80spec=sens_at_spec(s), brier=brier_score_loss(y, s) if k in PROB else np.nan))
T1 = pd.DataFrame(rows); print("TABLE 1 — out-of-fold performance, CDR>0 vs CDR=0, N =", N); print(T1.round(3).to_string(index=False))

# ---- Table 2: paired deltas with equivalence margin ----
MARGIN = 0.05; rows = []
for ref in ["nWBV only", "LR age+sex+eTIV+nWBV"]:
    for k in ORDER:
        if k == ref: continue
        v = np.array([roc_auc_score(y[b], M[k].values[b]) - roc_auc_score(y[b], M[ref].values[b]) for b in BI]); lo5, hi5 = np.percentile(v, [5, 95])
        rows.append(dict(reference=ref, model=k, delta_auroc=roc_auc_score(y, M[k]) - roc_auc_score(y, M[ref]), ci_lo=np.percentile(v, 2.5), ci_hi=np.percentile(v, 97.5),
                         excludes_0=bool(np.percentile(v, 2.5) > 0 or np.percentile(v, 97.5) < 0), equivalent_90ci_pm05=bool(lo5 > -MARGIN and hi5 < MARGIN)))
T2 = pd.DataFrame(rows); print("\nTABLE 2 — paired delta AUROC (same subjects, same folds)"); print(T2.round(3).to_string(index=False))

# ---- Figure 1: forest plot ----
fig, ax = plt.subplots(figsize=(8, 5)); yy = np.arange(len(T1))[::-1]
ax.errorbar(T1.auroc, yy, xerr=[T1.auroc - T1.ci_lo, T1.ci_hi - T1.auroc], fmt="o", capsize=3, color="C0"); ax.set_yticks(yy); ax.set_yticklabels(T1.model, fontsize=8)
ax.axvline(T1.auroc[T1.model == "nWBV only"].iloc[0], c="C3", ls="--", lw=.8, label="nWBV only"); ax.axvline(0.5, c="gray", lw=.5); ax.legend(fontsize=8, loc="lower right")
ax.set_xlabel("AUROC (95% subject-level bootstrap CI)"); ax.set_title("CDR>0 vs CDR=0, subject-level 5-fold CV"); plt.tight_layout(); plt.savefig(f"{OUT}/fig1_forest.png", dpi=200); plt.close()

# ---- Figure 2: full cohort vs age-matched ----
Am = pd.read_csv(f"{OUT}/roi_agematched.csv"); Am = Am[(Am.caliper == 2.0) & Am.ci_lo.notna()].set_index("model"); ks = [k for k in ORDER if k in Am.index]
fig, ax = plt.subplots(figsize=(8, 4.5)); xx = np.arange(len(ks)); full = T1.set_index("model").loc[ks]
ax.errorbar(xx - .1, full.auroc, yerr=[full.auroc - full.ci_lo, full.ci_hi - full.auroc], fmt="o", capsize=3, label="full cohort")
ax.errorbar(xx + .1, Am.loc[ks, "auroc"], yerr=[Am.loc[ks, "auroc"] - Am.loc[ks, "ci_lo"], Am.loc[ks, "ci_hi"] - Am.loc[ks, "auroc"]], fmt="s", capsize=3, label=f"age-matched (±2 y, n={int(Am.n.iloc[0])})")
ax.set_xticks(xx); ax.set_xticklabels(ks, rotation=40, ha="right", fontsize=7); ax.axhline(0.5, c="gray", lw=.5); ax.set_ylabel("AUROC"); ax.legend(fontsize=8)
ax.set_title("Effect of age matching"); plt.tight_layout(); plt.savefig(f"{OUT}/fig2_age_matched.png", dpi=200); plt.close()

# ---- Figure 3: ROC curves ----
fig, ax = plt.subplots(figsize=(5, 5))
for k in ["nWBV only", "LR age+sex+eTIV+nWBV", "CNN age-pretrained", "ROI MTL+vent (MRI-only)", ENS]:
    if k in M.columns: fpr, tpr, _ = roc_curve(y, M[k]); ax.plot(fpr, tpr, label=f"{k[:34]} ({roc_auc_score(y, M[k]):.3f})")
ax.plot([0, 1], [0, 1], c="gray", lw=.5); ax.set_xlabel("1 - specificity"); ax.set_ylabel("sensitivity"); ax.legend(fontsize=6, loc="lower right"); plt.tight_layout(); plt.savefig(f"{OUT}/fig3_roc.png", dpi=200); plt.close()

for nm, t in [("table1_main", T1), ("table2_paired_deltas", T2)]: t.to_csv(f"{OUT}/{nm}.csv", index=False)
for k in ["table1_main.csv", "table2_paired_deltas.csv", "fig1_forest.png", "fig2_age_matched.png", "fig3_roc.png"]: up(f"{OUT}/{k}", "paper")
