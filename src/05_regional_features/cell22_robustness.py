# ===== CELL 22 — Robustness of the hippocampal CSF-contact finding: fresh CV seeds, added value over nWBV and clinical covariates, age-matched / CDR 0.5 =====
# needs Cells 19 and 21 in the same session (FR, XP, POOL, fit_pred_sel, deltas3, age_match, imports) plus Cell 12 (D, y, N, BI, base, sens_at_spec)
FSTAR = "Hippocampus__r_gm_csf_contact"; assert FSTAR in FR.columns
XM = pd.concat([D[base].reset_index(drop=True), FR[[FSTAR]].reset_index(drop=True)], axis=1)
def lr_fit(cols, tr, te):
    m = make_pipeline(StandardScaler(), LogisticRegressionCV(Cs=np.logspace(-2, 1, 7), cv=5, scoring="roc_auc", class_weight="balanced", max_iter=5000)).fit(XM.loc[tr, cols], y[tr])
    return m.predict_proba(XM.loc[te, cols])[:, 1]
COV = "age+sex+eTIV+nWBV (clinical covariates)"
MOD = {"F* alone (fixed feature)": [FSTAR], "nWBV + F*": ["nwbv", FSTAR], COV: base, "age+sex+eTIV+nWBV + F*": base + [FSTAR]}
NR2 = 10; rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=NR2, random_state=2024); R5 = {k: np.zeros((NR2, N)) for k in list(MOD) + ["Top-1 in-fold selection"]}
for j, (tr_i, te_i) in enumerate(rskf.split(np.zeros(N), y)):
    r = j // 5; tr = np.zeros(N, bool); tr[tr_i] = True; te = ~tr
    for k, cols in MOD.items(): R5[k][r, te] = lr_fit(cols, tr, te)
    R5["Top-1 in-fold selection"][r, te] = fit_pred_sel(1, tr, te)
rows = [dict(model="nWBV only", auroc_mean=roc_auc_score(y, -D.nwbv.values), auroc_sd=0.0)]
for k, M in R5.items(): a = [roc_auc_score(y, M[r]) for r in range(NR2)]; rows.append(dict(model=k, auroc_mean=np.mean(a), auroc_sd=np.std(a)))
T5 = pd.DataFrame(rows); print(f"A. Fresh seeds: repeated 5-fold x {NR2} (F* = {FSTAR}, selected in 25/25 training sets in Cell 21)"); print(T5.round(3).to_string(index=False))
SC5 = {"nWBV only": -D.nwbv.values, **{k: M.mean(0) for k, M in R5.items()}}
D5 = deltas3(SC5, ["nWBV only", COV]); D5 = D5[((D5.reference == "nWBV only") & D5.model.isin(["F* alone (fixed feature)", "Top-1 in-fold selection", "nWBV + F*"])) | ((D5.reference == COV) & (D5.model == "age+sex+eTIV+nWBV + F*"))]
print("\nPaired delta AUROC (repeat-averaged scores):"); print(D5.round(3).to_string(index=False))

# ---- subsets and operating point ----
rng = np.random.default_rng(17); rows = []; subsets = {"all": np.arange(N), "CDR0.5_vs_0": np.where(D.cdr.isin([0, 0.5]).values)[0], "age-matched(2y)": age_match(2.0)}
show = ["nWBV only", "F* alone (fixed feature)", COV, "age+sex+eTIV+nWBV + F*"]
for sn, idx in subsets.items():
    ys = y[idx]; bs = [b for b in (rng.integers(0, len(idx), len(idx)) for _ in range(1000)) if len(np.unique(ys[b])) == 2]
    for k in show:
        s = SC5[k][idx]; v = [roc_auc_score(ys[b], s[b]) for b in bs]; fpr, tpr, _ = roc_curve(ys, s)
        rows.append(dict(subset=sn, n=len(idx), model=k, auroc=roc_auc_score(ys, s), ci_lo=np.percentile(v, 2.5), ci_hi=np.percentile(v, 97.5), sens_at_80spec=float(tpr[fpr <= 0.2].max())))
T6 = pd.DataFrame(rows); print("\nB. Subsets and operating point"); print(T6.round(3).to_string(index=False))

# ---- descriptive selection pressure (full-data univariate AUROC; optimistic, for context only) ----
R = rankdata(np.nan_to_num(XP), axis=0); n1 = int(y.sum()); n0 = N - n1; au = pd.Series((R[y == 1].sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0), index=POOL); au = (au - .5).abs() + .5
print(f"\nC. Full-data univariate AUROC of the 96 pool features (optimistic): top 8, and how many exceed nWBV ({roc_auc_score(y, -D.nwbv.values):.3f})"); print(au.sort_values(ascending=False).head(8).round(3).to_string())
print("features above nWBV:", int((au > roc_auc_score(y, -D.nwbv.values)).sum()), "of", len(au))
for nm, t in [("robust_fresh_seeds", T5), ("robust_deltas", D5), ("robust_subsets", T6)]: t.to_csv(f"{OUT}/{nm}.csv", index=False); up(f"{OUT}/{nm}.csv", "roi")
