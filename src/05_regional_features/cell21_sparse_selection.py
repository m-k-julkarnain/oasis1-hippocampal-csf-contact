# ===== CELL 21 — Sparse model: in-fold univariate feature selection (top-k of 96 ROI features) vs nWBV; pre-specified primary k=3 =====
# needs Cell 19 in the same session (FR, vol, rd, RN, deltas3, S3, age_match, imports) plus Cell 12 (D, y, N, BI, sens_at_spec)
from scipy.stats import rankdata
POOL = vol(RN) + rd(RN); XP = FR[POOL].values.astype(float); KS = [1, 3, 5]; PRIMARY_K = 3; SELCOUNT = {k: pd.Series(0, index=POOL) for k in KS}
def select_topk(Xtr, ytr, k):
    R = rankdata(Xtr, axis=0); n1 = int(ytr.sum()); n0 = len(ytr) - n1; au = (R[ytr == 1].sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0)
    return np.argsort(-np.abs(au - 0.5))[:k]
def fit_pred_sel(k, tr, te, count=False):
    imp = SimpleImputer(strategy="median").fit(XP[tr]); Xtr, Xte = imp.transform(XP[tr]), imp.transform(XP[te]); idx = select_topk(Xtr, y[tr], k)
    if count: SELCOUNT[k].iloc[idx] += 1
    m = make_pipeline(StandardScaler(), LogisticRegressionCV(Cs=np.logspace(-2, 1, 7), cv=5, scoring="roc_auc", class_weight="balanced", max_iter=5000)).fit(Xtr[:, idx], y[tr])
    return m.predict_proba(Xte[:, idx])[:, 1]
NAME = {k: f"Top-{k} of 96 (in-fold selection)" + (" (PRIMARY)" if k == PRIMARY_K else "") for k in KS}

# ---- A. identical folds ----
S4 = {"nWBV only": S3["nWBV only"], "VOL hippocampus+amygdala (post-hoc, Cell 17)": S3["VOL hippocampus+amygdala (post-hoc, Cell 17)"]}
for k in KS:
    s = np.zeros(N)
    for f in range(5): te = (D.fold == f).values; s[te] = fit_pred_sel(k, ~te, te, count=(k == PRIMARY_K))
    S4[NAME[k]] = s
rows = []
for k, s in S4.items():
    v = [roc_auc_score(y[i], s[i]) for i in BI]; rows.append(dict(model=k, auroc=roc_auc_score(y, s), ci_lo=np.percentile(v, 2.5), ci_hi=np.percentile(v, 97.5), avg_precision=average_precision_score(y, s), sens_at_80spec=sens_at_spec(s)))
TA4 = pd.DataFrame(rows); print("A. Identical folds, N =", N); print(TA4.round(3).to_string(index=False))
DA4 = deltas3(S4, ["nWBV only"]); print("\nPaired delta AUROC vs nWBV:"); print(DA4.round(3).to_string(index=False))

# ---- B. repeated 5x5 CV (also gives selection frequencies for the primary model) ----
for k in KS: SELCOUNT[k][:] = 0
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=NREP, random_state=7); RS4 = {k: np.zeros((NREP, N)) for k in KS}
for j, (tr_i, te_i) in enumerate(rskf.split(np.zeros(N), y)):
    r = j // 5; tr = np.zeros(N, bool); tr[tr_i] = True; te = ~tr
    for k in KS: RS4[k][r, te] = fit_pred_sel(k, tr, te, count=(k == PRIMARY_K))
rows = [dict(model="nWBV only", auroc_mean=roc_auc_score(y, -D.nwbv.values), auroc_sd=0.0)]
for k, M in RS4.items(): a = [roc_auc_score(y, M[r]) for r in range(NREP)]; rows.append(dict(model=NAME[k], auroc_mean=np.mean(a), auroc_sd=np.std(a)))
TB4 = pd.DataFrame(rows); print(f"\nB. Repeated 5-fold x {NREP}"); print(TB4.round(3).to_string(index=False))
SC4 = {"nWBV only": -D.nwbv.values, **{NAME[k]: M.mean(0) for k, M in RS4.items()}}; DB4 = deltas3(SC4, ["nWBV only"]); print("\nPaired delta AUROC vs nWBV (repeat-averaged scores):"); print(DB4.round(3).to_string(index=False))
sel = (SELCOUNT[PRIMARY_K] / (5 * NREP)).sort_values(ascending=False); print(f"\nSelection frequency of features in the primary model (fraction of the {5*NREP} training sets):"); print(sel[sel > 0].round(2).head(10).to_string())

# ---- C. CDR 0.5 vs 0 and age-matched subsets ----
rng = np.random.default_rng(13); rows = []; subsets = {"CDR0.5_vs_0": np.where(D.cdr.isin([0, 0.5]).values)[0], "age-matched(2y)": age_match(2.0)}
for sn, idx in subsets.items():
    ys = y[idx]; bs = [b for b in (rng.integers(0, len(idx), len(idx)) for _ in range(1000)) if len(np.unique(ys[b])) == 2]
    for k, s in S4.items():
        v = [roc_auc_score(ys[b], s[idx][b]) for b in bs]; rows.append(dict(subset=sn, n=len(idx), model=k, auroc=roc_auc_score(ys, s[idx]), ci_lo=np.percentile(v, 2.5), ci_hi=np.percentile(v, 97.5)))
TC4 = pd.DataFrame(rows); print("\nC. Subsets"); print(TC4.round(3).to_string(index=False))
P = NAME[PRIMARY_K]; pa = DA4[DA4.model == P].iloc[0]; pb = DB4[DB4.model == P].iloc[0]
print(f"\nPRIMARY (top-{PRIMARY_K}) vs nWBV: identical folds {pa.delta_auroc:+.3f} (95% CI {pa.ci_lo:+.3f} to {pa.ci_hi:+.3f}) | repeated CV {pb.delta_auroc:+.3f} (95% CI {pb.ci_lo:+.3f} to {pb.ci_hi:+.3f})")
print("Models tried in this cell: 3 (report all). Success criterion fixed in advance: both CIs exclude 0.")
for nm, t in [("sparse_identical_folds", TA4), ("sparse_paired_deltas", DA4), ("sparse_repeatedcv", TB4), ("sparse_repeatedcv_deltas", DB4), ("sparse_subsets", TC4)]: t.to_csv(f"{OUT}/{nm}.csv", index=False); up(f"{OUT}/{nm}.csv", "roi")
sel.to_csv(f"{OUT}/sparse_selection_frequency.csv"); up(f"{OUT}/sparse_selection_frequency.csv", "roi")
