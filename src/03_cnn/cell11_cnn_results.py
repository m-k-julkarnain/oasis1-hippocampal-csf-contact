# ===== CELL 11 — CNN results: CDR AUROC (with CIs), brain-age gap association, comparison with earlier baselines =====
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
rng = np.random.default_rng(0)
oof = pd.concat(all_rows, ignore_index=True).merge(lab, on="subject_id"); y = oof.cdr_binary.astype(int).values; N = len(oof)
S = {"CNN age-pretrained": oof.p_agepre.values, "CNN scratch": oof.p_scratch.values, "nWBV only": -oof.nwbv.values,
     "CNN BAG (raw)": oof.bag_raw.values, "CNN BAG (corrected)": oof.bag_corr.values}
BI = [rng.integers(0, N, N) for _ in range(2000)]; BI = [i for i in BI if len(np.unique(y[i])) == 2]
def auc_ci(s): v = [roc_auc_score(y[i], s[i]) for i in BI]; return roc_auc_score(y, s), np.percentile(v, 2.5), np.percentile(v, 97.5)
def diff_ci(s1, s2): v = [roc_auc_score(y[i], s1[i]) - roc_auc_score(y[i], s2[i]) for i in BI]; return roc_auc_score(y, s1) - roc_auc_score(y, s2), np.percentile(v, 2.5), np.percentile(v, 97.5)
rows = [dict(model=k, auroc=a, ci_lo=l, ci_hi=h, fold_mean=np.mean([roc_auc_score(y[oof.fold == f], s[oof.fold == f]) for f in range(5)]),
             fold_sd=np.std([roc_auc_score(y[oof.fold == f], s[oof.fold == f]) for f in range(5)])) for k, s in S.items() for a, l, h in [auc_ci(s)]]
T = pd.DataFrame(rows); print(T.round(3).to_string(index=False))
for a, b in [("CNN age-pretrained", "CNN scratch"), ("CNN age-pretrained", "nWBV only")]:
    d, l, h = diff_ci(S[a], S[b]); print(f"paired delta AUROC {a} - {b}: {d:+.3f} [{l:+.3f}, {h:+.3f}]")
n0 = oof[oof.cdr == 0]; print(f"\nbrain-age CNN: held-out CDR0 MAE {np.abs(n0.age_pred - n0.age).mean():.2f}y | mean BAG raw CDR0 {n0.bag_raw.mean():+.2f} CDR>0 {oof[oof.cdr>0].bag_raw.mean():+.2f}")
cov = oof[["age", "sex_f", "etiv"]].values
def resid(x, C): C = np.column_stack([np.ones(len(C)), C]); return x - C @ np.linalg.lstsq(C, x, rcond=None)[0]
def pspear(x, z): return np.corrcoef(resid(pd.Series(x).rank().values, cov), resid(pd.Series(z).rank().values, cov))[0, 1]
print(f"MMSE spearman: BAG raw {spearmanr(oof.bag_raw, oof.mmse)[0]:+.3f} (partial {pspear(oof.bag_raw, oof.mmse):+.3f}) | nWBV {spearmanr(oof.nwbv, oof.mmse)[0]:+.3f} (partial {pspear(oof.nwbv, oof.mmse):+.3f}) | CNN agepre prob {spearmanr(oof.p_agepre, oof.mmse)[0]:+.3f}")
def cv_probs(cols, d, n_rep=5):
    rs = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_rep, random_state=1); per = np.zeros((n_rep, len(d)))
    for sp, (a, b) in enumerate(rs.split(d, d.cdr_binary)):
        m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)).fit(d.iloc[a][cols], d.cdr_binary.iloc[a])
        per[sp // 5, b] = m.predict_proba(d.iloc[b][cols])[:, 1]
    return np.mean([roc_auc_score(d.cdr_binary, per[r]) for r in range(n_rep)]), per.mean(0)
d = oof.assign(cdr_binary=y); base = ["age", "sex_f", "etiv", "nwbv"]
m0, p0 = cv_probs(base, d); m1, p1 = cv_probs(base + ["bag_raw"], d)
dd = [roc_auc_score(y[i], p1[i]) - roc_auc_score(y[i], p0[i]) for i in BI]
print(f"base(age,sex,eTIV,nWBV) CV AUROC {m0:.3f} -> +CNN BAG {m1:.3f} | delta {m1-m0:+.3f} [{np.percentile(dd,2.5):+.3f}, {np.percentile(dd,97.5):+.3f}]")
try:
    R2 = pd.read_csv(f"{OUT}/brainage_blocks_results.csv"); print(f"reference (earlier cells): nWBV-only {R2.auroc_nWBV_only[0]:.3f} | block-feature direct MRI-only CV AUROC {R2.cv_auroc_blocks_direct[0]:.3f}")
except Exception: pass
T.to_csv(f"{OUT}/cnn_results.csv", index=False); oof.to_csv(f"{OUT}/cnn_oof_all.csv", index=False)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
for k in ["CNN age-pretrained", "CNN scratch", "nWBV only"]:
    fpr, tpr, _ = roc_curve(y, S[k]); ax[0].plot(fpr, tpr, label=f"{k} ({roc_auc_score(y, S[k]):.2f})")
ax[0].plot([0, 1], [0, 1], "k--"); ax[0].legend(); ax[0].set_title("Out-of-fold ROC (subject-level 5-fold)")
cs = sorted(oof.cdr.unique()); ax[1].boxplot([oof[oof.cdr == c].bag_raw for c in cs]); ax[1].set_xticklabels([f"CDR {c}" for c in cs]); ax[1].axhline(0, c="k", lw=.5); ax[1].set_title("CNN brain-age gap (raw, years)")
plt.tight_layout(); plt.savefig(f"{OUT}/cnn_figure.png", dpi=120); plt.close()
for k in ["cnn_results.csv", "cnn_oof_all.csv", "cnn_figure.png"]: up(f"{OUT}/{k}", "cnn")
