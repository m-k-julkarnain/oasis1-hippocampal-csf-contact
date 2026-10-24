# ===== CELL 4 — Does brain-age gap add information? CDR AUROC, MMSE correlation, incremental value =====
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
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
    rx, ry = resid(pd.Series(x).rank().values, covs), resid(pd.Series(y).rank().values, covs)
    return np.corrcoef(rx, ry)[0, 1]

def cv_probs(cols, d, n_rep=5):
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_rep, random_state=1); per = np.zeros((n_rep, len(d)))
    for sp, (tr, te) in enumerate(rskf.split(d, d.cdr_binary)):
        m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000))
        m.fit(d.iloc[tr][cols], d.cdr_binary.iloc[tr]); per[sp // 5, te] = m.predict_proba(d.iloc[te][cols])[:, 1]
    aucs = [roc_auc_score(d.cdr_binary, per[r]) for r in range(n_rep)]
    return float(np.mean(aucs)), float(np.std(aucs)), per.mean(0)

rows = []
for (v, mn), d in BAG.items():
    d = d.copy(); d["cdr_binary"] = d.cdr_binary.astype(int); cov = d[["age", "sex_f", "etiv"]].values
    auc_bag = roc_auc_score(d.cdr_binary, d.bag_corr); ci_bag = boot_ci(lambda s: roc_auc_score(s.cdr_binary, s.bag_corr), d)
    auc_nwbv = roc_auc_score(d.cdr_binary, -d.nwbv)
    rho, p = spearmanr(d.bag_corr, d.mmse); prho = partial_spearman(d.bag_corr, d.mmse, cov)
    base = ["age", "sex_f", "etiv", "nwbv"]
    m0, s0, p0 = cv_probs(base, d); m1, s1, p1 = cv_probs(base + ["bag_corr"], d)
    m2, s2, _ = cv_probs(["bag_corr"], d); m3, s3, _ = cv_probs(FEATS + ["age", "sex_f"], d)
    dd = d.assign(p0=p0, p1=p1)
    dci = boot_ci(lambda s: roc_auc_score(s.cdr_binary, s.p1) - roc_auc_score(s.cdr_binary, s.p0), dd)
    rows.append(dict(age_model=f"{v}/{mn}", n=len(d), auroc_BAG=auc_bag, BAG_ci_lo=ci_bag[0], BAG_ci_hi=ci_bag[1], auroc_nWBV_only=auc_nwbv,
                     mmse_spearman=rho, mmse_partial_spearman=prho, cv_auroc_base=m0, cv_auroc_base_plus_BAG=m1, delta_auroc=m1 - m0,
                     delta_ci_lo=dci[0], delta_ci_hi=dci[1], cv_auroc_BAG_only=m2, cv_auroc_all_volumetric=m3))
R = pd.DataFrame(rows); R.to_csv(f"{OUT}/brainage_results.csv", index=False)
print(R.round(3).T.to_string())
print("\nbase = age+sex+eTIV+nWBV. delta_auroc CI crossing 0 => BAG adds no reliable information beyond volume.")

# figure for the first configuration
v, mn = list(BAG)[0]; d = BAG[(v, mn)]
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
cs = sorted(d.cdr.unique()); ax[0].boxplot([d[d.cdr == c].bag_corr for c in cs])
ax[0].set_xticklabels([f"CDR {c}\n(n={int((d.cdr==c).sum())})" for c in cs])
ax[0].axhline(0, c="k", lw=.5); ax[0].set_ylabel("bias-corrected brain-age gap (years)"); ax[0].set_title(f"BAG by CDR ({v}/{mn})")
ax[1].scatter(d.bag_corr, d.mmse, c=d.cdr_binary, cmap="coolwarm", s=14); ax[1].set_xlabel("brain-age gap (years)"); ax[1].set_ylabel("MMSE"); ax[1].set_title("BAG vs MMSE")
plt.tight_layout(); plt.savefig(f"{OUT}/brainage_figure.png", dpi=120); plt.close()
for k in ["brainage_results.csv", "brainage_figure.png", "features.csv"]: up(f"{OUT}/{k}", "brainage")
