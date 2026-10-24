# ===== CELL 7 — Association analysis for block-feature brain age (+ direct MRI-only CDR classifier) =====
from sklearn.decomposition import PCA
rng = np.random.default_rng(0)
def boot_ci(fn, d, n=1000):
    v = []
    for _ in range(n):
        s = d.iloc[rng.integers(0, len(d), len(d))]
        if s.cdr_binary.nunique() == 2: v.append(fn(s))
    return np.percentile(v, [2.5, 97.5])
def resid(x, C):
    C = np.column_stack([np.ones(len(C)), C]); return x - C @ np.linalg.lstsq(C, x, rcond=None)[0]
def partial_spearman(x, y, covs):
    return np.corrcoef(resid(pd.Series(x).rank().values, covs), resid(pd.Series(y).rank().values, covs))[0, 1]
def cv_probs(cols, d, n_rep=5):
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_rep, random_state=1); per = np.zeros((n_rep, len(d)))
    for sp, (tr, te) in enumerate(rskf.split(d, d.cdr_binary)):
        m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000))
        m.fit(d.iloc[tr][cols], d.cdr_binary.iloc[tr]); per[sp // 5, te] = m.predict_proba(d.iloc[te][cols])[:, 1]
    return float(np.mean([roc_auc_score(d.cdr_binary, per[r]) for r in range(n_rep)])), per.mean(0)
def cv_blocks_direct(d, n_rep=3):          # MRI-only classifier on block features (no age/sex)
    Xd = XB[[SID[s] for s in d.subject_id]]; y = d.cdr_binary.values
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_rep, random_state=2); per = np.zeros((n_rep, len(d)))
    for sp, (tr, te) in enumerate(rskf.split(Xd, y)):
        m = make_pipeline(StandardScaler(), PCA(n_components=40, random_state=0), LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000))
        m.fit(Xd[tr], y[tr]); per[sp // 5, te] = m.predict_proba(Xd[te])[:, 1]
    return float(np.mean([roc_auc_score(y, per[r]) for r in range(n_rep)]))

rows = []
for (v, mn), d in BAG_BLK.items():
    d = d.copy(); d["cdr_binary"] = d.cdr_binary.astype(int); cov = d[["age", "sex_f", "etiv"]].values
    ci = boot_ci(lambda s: roc_auc_score(s.cdr_binary, s.bag_corr), d); base = ["age", "sex_f", "etiv", "nwbv"]
    m0, p0 = cv_probs(base, d); m1, p1 = cv_probs(base + ["bag_corr"], d); mb, _ = cv_probs(["bag_corr"], d)
    dci = boot_ci(lambda s: roc_auc_score(s.cdr_binary, s.p1) - roc_auc_score(s.cdr_binary, s.p0), d.assign(p0=p0, p1=p1))
    rows.append(dict(age_model=f"{v}/{mn}", n=len(d), auroc_BAG=roc_auc_score(d.cdr_binary, d.bag_corr), BAG_ci_lo=ci[0], BAG_ci_hi=ci[1],
                     auroc_nWBV_only=roc_auc_score(d.cdr_binary, -d.nwbv),
                     mmse_spearman_BAG=spearmanr(d.bag_corr, d.mmse)[0], mmse_partial_BAG=partial_spearman(d.bag_corr, d.mmse, cov),
                     mmse_spearman_nWBV=spearmanr(d.nwbv, d.mmse)[0], mmse_partial_nWBV=partial_spearman(d.nwbv, d.mmse, d[["age", "sex_f", "etiv"]].values),
                     cv_auroc_base=m0, cv_auroc_base_plus_BAG=m1, delta_auroc=m1 - m0, delta_ci_lo=dci[0], delta_ci_hi=dci[1],
                     cv_auroc_BAG_only=mb, cv_auroc_blocks_direct=cv_blocks_direct(d)))
R2 = pd.DataFrame(rows); R2.to_csv(f"{OUT}/brainage_blocks_results.csv", index=False); print(R2.round(3).T.to_string())
up(f"{OUT}/brainage_blocks_results.csv", "brainage")
