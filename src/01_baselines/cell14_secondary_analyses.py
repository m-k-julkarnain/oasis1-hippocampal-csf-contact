# ===== CELL 14 — Secondary analyses on identical folds: stage-stratified AUROC and MMSE regression =====
# needs the variables from Cell 12 (D, S, y, N, NAMES, XD, base, allv) in the same session
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr
rng = np.random.default_rng(1)

# ---- A. Which stages are the models actually good at?  (CDR 0.5 vs 0, and CDR>=1 vs 0)
stages = {"CDR0.5_vs_0": D.cdr.isin([0, 0.5]).values, "CDR>=1_vs_0": D.cdr.isin([0, 1, 2]).values}; rows = []
for sn, msk in stages.items():
    idx = np.where(msk)[0]; ys = y[idx]; bs = [rng.integers(0, len(idx), len(idx)) for _ in range(1000)]; bs = [b for b in bs if len(np.unique(ys[b])) == 2]
    for k in NAMES:
        s = S[k][idx]; v = [roc_auc_score(ys[b], s[b]) for b in bs]
        rows.append(dict(task=sn, n_pos=int(ys.sum()), n_neg=int((1 - ys).sum()), model=k, auroc=roc_auc_score(ys, s), ci_lo=np.percentile(v, 2.5), ci_hi=np.percentile(v, 97.5)))
A = pd.DataFrame(rows); print(A.round(3).to_string(index=False)); A.to_csv(f"{OUT}/stage_stratified_auroc.csv", index=False)

# ---- B. MMSE regression (same folds). MMSE is skewed with a ceiling at 30, so report MAE/RMSE/Spearman and a mean-only baseline
mm = D.mmse.values.astype(float); names = ["mean baseline", "nWBV (linear)", "age+sex+eTIV+nWBV (linear)", "all volumetric+age+sex (ridge)", "Blocks (ridge)"]
P = {k: np.zeros(N) for k in names}
for f in range(5):
    te = (D.fold == f).values; tr = ~te
    P["mean baseline"][te] = mm[tr].mean()
    P["nWBV (linear)"][te] = LinearRegression().fit(D.loc[tr, ["nwbv"]], mm[tr]).predict(D.loc[te, ["nwbv"]])
    P["age+sex+eTIV+nWBV (linear)"][te] = LinearRegression().fit(D.loc[tr, base], mm[tr]).predict(D.loc[te, base])
    P["all volumetric+age+sex (ridge)"][te] = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 12))).fit(D.loc[tr, allv], mm[tr]).predict(D.loc[te, allv])
    P["Blocks (ridge)"][te] = RidgeCV(alphas=np.logspace(0, 5, 11)).fit(XD[tr], mm[tr]).predict(XD[te])
BM = [rng.integers(0, N, N) for _ in range(1000)]; rows = []
for k in names:
    e = np.abs(P[k] - mm); v = [e[b].mean() for b in BM]
    rows.append(dict(model=k, n=N, mae=e.mean(), mae_ci_lo=np.percentile(v, 2.5), mae_ci_hi=np.percentile(v, 97.5), rmse=np.sqrt(((P[k] - mm) ** 2).mean()),
                     spearman=spearmanr(P[k], mm)[0] if k != "mean baseline" else np.nan))
rows.append(dict(model="CNN age-pretrained CDR prob (Spearman only, not a regressor)", n=N, spearman=spearmanr(S["CNN age-pretrained"], mm)[0]))
B = pd.DataFrame(rows); print("\nMMSE regression:"); print(B.round(3).to_string(index=False)); B.to_csv(f"{OUT}/mmse_regression.csv", index=False)
for k in ["stage_stratified_auroc.csv", "mmse_regression.csv"]: up(f"{OUT}/{k}", "benchmark")
