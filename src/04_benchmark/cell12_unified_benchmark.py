# ===== CELL 12 — Unified benchmark on IDENTICAL folds (paired bootstrap) =====
# needs Cell 1, Cell 3 (lab, unl, FEATS, imports), Cell 5 (blocks.npz) and Cell 10 (cnn_fold*.csv, which fixes the 5 folds)
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, brier_score_loss, roc_curve
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
rng = np.random.default_rng(0)
cnn = pd.concat([pd.read_csv(f"{OUT}/cnn_fold{f}.csv") for f in range(5)], ignore_index=True)
D = lab.merge(cnn, on="subject_id").reset_index(drop=True); y = D.cdr_binary.astype(int).values; N = len(D)
z = np.load(f"{OUT}/blocks.npz", allow_pickle=True); XB = z["X"].astype(np.float32); SID = {s: i for i, s in enumerate(z["sids"])}
XD = XB[[SID[s] for s in D.subject_id]]
base = ["age", "sex_f", "etiv", "nwbv"]; allv = FEATS + ["age", "sex_f"]
NAMES = ["nWBV only", "LR age+sex+eTIV+nWBV", "LR all volumetric+age+sex", "Blocks MRI-only (PCA+LR)", "Block-ridge BAG (raw)",
         "CNN scratch", "CNN age-pretrained", "CNN BAG (raw)"]
PROB = {"LR age+sex+eTIV+nWBV", "LR all volumetric+age+sex", "Blocks MRI-only (PCA+LR)", "CNN scratch", "CNN age-pretrained"}
S = {k: np.zeros(N) for k in NAMES}
def lr(cols, tr, te):
    m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)).fit(D.loc[tr, cols], y[tr])
    return m.predict_proba(D.loc[te, cols])[:, 1]
for f in range(5):
    te = (D.fold == f).values; tr = ~te
    S["nWBV only"][te] = -D.nwbv.values[te]
    S["LR age+sex+eTIV+nWBV"][te] = lr(base, tr, te); S["LR all volumetric+age+sex"][te] = lr(allv, tr, te)
    m = make_pipeline(StandardScaler(), PCA(n_components=40, random_state=0), LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000)).fit(XD[tr], y[tr])
    S["Blocks MRI-only (PCA+LR)"][te] = m.predict_proba(XD[te])[:, 1]
    nm = pd.concat([D[tr & (D.cdr == 0).values][["subject_id", "age"]], unl[["subject_id", "age"]]], ignore_index=True)
    rm = RidgeCV(alphas=np.logspace(0, 5, 11)).fit(XB[[SID[s] for s in nm.subject_id]], nm.age.values)
    S["Block-ridge BAG (raw)"][te] = rm.predict(XD[te]) - D.age.values[te]
    S["CNN scratch"][te] = D.p_scratch.values[te]; S["CNN age-pretrained"][te] = D.p_agepre.values[te]; S["CNN BAG (raw)"][te] = D.bag_raw.values[te]

BI = [rng.integers(0, N, N) for _ in range(2000)]; BI = [i for i in BI if len(np.unique(y[i])) == 2]
def sens_at_spec(s, spec=0.8): fpr, tpr, _ = roc_curve(y, s); return float(tpr[fpr <= 1 - spec].max())
rows = []
for k in NAMES:
    s = S[k]; v = [roc_auc_score(y[i], s[i]) for i in BI]; fa = [roc_auc_score(y[D.fold == f], s[(D.fold == f).values]) for f in range(5)]
    rows.append(dict(model=k, auroc=roc_auc_score(y, s), ci_lo=np.percentile(v, 2.5), ci_hi=np.percentile(v, 97.5), avg_precision=average_precision_score(y, s),
                     sens_at_80spec=sens_at_spec(s), brier=brier_score_loss(y, s) if k in PROB else np.nan, fold_mean=np.mean(fa), fold_sd=np.std(fa)))
T = pd.DataFrame(rows); print(T.round(3).to_string(index=False))
drows = []
for ref in ["LR age+sex+eTIV+nWBV", "nWBV only"]:
    for k in NAMES:
        if k == ref: continue
        v = [roc_auc_score(y[i], S[k][i]) - roc_auc_score(y[i], S[ref][i]) for i in BI]
        drows.append(dict(reference=ref, model=k, delta_auroc=roc_auc_score(y, S[k]) - roc_auc_score(y, S[ref]), ci_lo=np.percentile(v, 2.5), ci_hi=np.percentile(v, 97.5)))
Dl = pd.DataFrame(drows); Dl["excludes_0"] = (Dl.ci_lo > 0) | (Dl.ci_hi < 0); print("\nPaired delta AUROC (same subjects, same folds):"); print(Dl.round(3).to_string(index=False))
T.to_csv(f"{OUT}/unified_benchmark.csv", index=False); Dl.to_csv(f"{OUT}/unified_paired_deltas.csv", index=False)
pd.DataFrame(S).assign(subject_id=D.subject_id, fold=D.fold, cdr_binary=y).to_csv(f"{OUT}/unified_oof_scores.csv", index=False)
fig, ax = plt.subplots(figsize=(7, 4.5)); yy = np.arange(len(T))[::-1]
ax.errorbar(T.auroc, yy, xerr=[T.auroc - T.ci_lo, T.ci_hi - T.auroc], fmt="o", capsize=3); ax.set_yticks(yy); ax.set_yticklabels(T.model, fontsize=8)
ax.axvline(0.5, c="gray", lw=.5); ax.set_xlabel("out-of-fold AUROC (95% subject bootstrap CI)"); ax.set_title("CDR>0 vs CDR=0, identical subject-level folds")
plt.tight_layout(); plt.savefig(f"{OUT}/unified_benchmark.png", dpi=120); plt.close()
for k in ["unified_benchmark.csv", "unified_paired_deltas.csv", "unified_oof_scores.csv", "unified_benchmark.png"]: up(f"{OUT}/{k}", "benchmark")
