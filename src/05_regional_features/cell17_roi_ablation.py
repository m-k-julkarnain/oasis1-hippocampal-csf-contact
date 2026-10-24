# ===== CELL 17 — What drives the ROI gain?  ROI-group ablation, age-matched sensitivity analysis, ROI brain-age gap =====
# needs Cells 3, 12 and 16 in the same session (D, S2, RS, y, N, BI, base, XR, MTL, sets, roi, ALL, unl, fit_pred, deltas, sens_at_spec, MARGIN, imports)
# ---- A. ROI-group ablation (repeated 5x5 CV, same splits as Cell 16) ----
GROUPS = {"Hippocampus": ["Hippocampus"], "Amygdala": ["Amygdala"], "Parahippocampal": [n for n in sets["apriori"] if n.startswith("Parahippocampal")],
          "Temporal pole": ["Temporal_Pole"], "Lateral ventricle": ["Lateral_Ventricle"]}
def cols_of(rois): return [c for c in MTL if c.split("__")[0] in rois]
allr = sum(GROUPS.values(), [])
ABL = {f"only {g}": cols_of(r) for g, r in GROUPS.items()}
ABL.update({"Hippocampus + amygdala": cols_of(GROUPS["Hippocampus"] + GROUPS["Amygdala"]),
            "MTL without ventricle": cols_of([r for r in allr if r != "Lateral_Ventricle"]),
            "MTL without hippocampus": cols_of([r for r in allr if r != "Hippocampus"]),
            "MTL without amygdala": cols_of([r for r in allr if r != "Amygdala"])})
ABL = {k: v for k, v in ABL.items() if len(v) > 0}
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=NREP, random_state=7); RA = {k: np.zeros((NREP, N)) for k in ABL}
for j, (tr_i, te_i) in enumerate(rskf.split(np.zeros(N), y)):
    r = j // 5; tr = np.zeros(N, bool); tr[tr_i] = True; te = ~tr
    for k, cols in ABL.items(): RA[k][r, te] = fit_pred(cols, 0.1, tr, te)
rows = [dict(model="nWBV only", n_features=1, auroc_mean=roc_auc_score(y, -D.nwbv.values), auroc_sd=0.0),
        dict(model="MTL+vent full (Cell 16)", n_features=len(MTL), auroc_mean=np.mean([roc_auc_score(y, RS["ROI MTL+vent (MRI-only)"][r]) for r in range(NREP)]),
             auroc_sd=np.std([roc_auc_score(y, RS["ROI MTL+vent (MRI-only)"][r]) for r in range(NREP)]))]
for k, M in RA.items(): a = [roc_auc_score(y, M[r]) for r in range(NREP)]; rows.append(dict(model=k, n_features=len(ABL[k]), auroc_mean=np.mean(a), auroc_sd=np.std(a)))
TAb = pd.DataFrame(rows); print("A. ROI-group ablation (MRI-only logistic regression, repeated 5x5 CV)"); print(TAb.round(3).to_string(index=False))
SCa = {"nWBV only": -D.nwbv.values, "MTL+vent full (Cell 16)": RS["ROI MTL+vent (MRI-only)"].mean(0), **{k: M.mean(0) for k, M in RA.items()}}
DAb = deltas(SCa, ["nWBV only", "MTL+vent full (Cell 16)"]); DAb["excludes_0"] = (DAb.ci_lo > 0) | (DAb.ci_hi < 0)
print("\nPaired delta AUROC (repeat-averaged scores):"); print(DAb.round(3).to_string(index=False))

# ---- B. age-matched sensitivity analysis (out-of-fold scores from Cell 16, evaluated on an age-matched subset) ----
age = D.age.values
def age_match(caliper):
    pos = np.where(y == 1)[0]; neg = list(np.where(y == 0)[0]); keep = []
    for i in pos[np.argsort(age[pos])]:
        if not neg: break
        d = np.abs(age[neg] - age[i]); j = int(np.argmin(d))
        if d[j] <= caliper: keep += [i, neg.pop(j)]
    return np.array(sorted(keep))
rng = np.random.default_rng(5); rows = []
for cal in [2.0, 4.0]:
    idx = age_match(cal); ys = y[idx]; bs = [b for b in (rng.integers(0, len(idx), len(idx)) for _ in range(1000)) if len(np.unique(ys[b])) == 2]
    info = f"caliper {cal:g}y: {int((ys==1).sum())} pairs | age CDR>0 {age[idx][ys==1].mean():.1f} vs CDR0 {age[idx][ys==0].mean():.1f}"; print("\nB. Age-matched,", info)
    for k, s in S2.items():
        v = [roc_auc_score(ys[b], s[idx][b]) for b in bs]
        rows.append(dict(caliper=cal, n=len(idx), model=k, auroc=roc_auc_score(ys, s[idx]), ci_lo=np.percentile(v, 2.5), ci_hi=np.percentile(v, 97.5)))
    rows.append(dict(caliper=cal, n=len(idx), model="age alone", auroc=roc_auc_score(ys, age[idx]), ci_lo=np.nan, ci_hi=np.nan))
TBm = pd.DataFrame(rows); print(TBm.round(3).to_string(index=False))

# ---- C. ROI-based brain-age gap (ridge on all ROI features, trained on normals + young presumed-normal, identical folds) ----
Xall = roi.set_index("subject_id")[ALL]; Xd = Xall.loc[D.subject_id].values; bag = np.zeros(N); maes = []; norm = (D.cdr == 0).values
for f in range(5):
    te = (D.fold == f).values; tr = ~te; trn = tr & norm
    ids = list(D.subject_id[trn]) + list(unl.subject_id); ages = np.r_[D.age.values[trn], unl.age.values]
    m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(0, 4, 9))).fit(Xall.loc[ids].values, ages)
    bag[te] = m.predict(Xd[te]) - D.age.values[te]; nte = te & norm; maes.append(mean_absolute_error(D.age.values[nte], m.predict(Xd[nte])))
v = [roc_auc_score(y[i], bag[i]) for i in BI]; mm = D.mmse.notna().values
TCb = pd.DataFrame([dict(model="ROI brain-age gap (raw)", auroc=roc_auc_score(y, bag), ci_lo=np.percentile(v, 2.5), ci_hi=np.percentile(v, 97.5), mae_held_out_normals=np.mean(maes),
                         spearman_mmse=spearmanr(bag[mm], D.mmse[mm])[0])])
print("\nC. ROI brain-age gap vs CDR>0"); print(TCb.round(3).to_string(index=False))
dv = np.array([roc_auc_score(y[i], bag[i]) - roc_auc_score(y[i], -D.nwbv.values[i]) for i in BI]); print(f"delta AUROC vs nWBV: {roc_auc_score(y, bag) - roc_auc_score(y, -D.nwbv.values):+.3f} (95% CI {np.percentile(dv, 2.5):+.3f} to {np.percentile(dv, 97.5):+.3f})")
for nm, t in [("roi_ablation", TAb), ("roi_ablation_deltas", DAb), ("roi_agematched", TBm), ("roi_bag", TCb)]: t.to_csv(f"{OUT}/{nm}.csv", index=False); up(f"{OUT}/{nm}.csv", "roi")
