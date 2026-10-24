# ===== CELL 19 — Radiomics-style regional features (intensity, texture, CSF-contact) vs nWBV; pre-specified primary test =====
# needs Cell 1, Cell 3 (imports, unl), Cell 12 (D, y, N, BI, sens_at_spec), Cell 15 (rid, names, apriori, roi, SIDS, sub_dirs, load_seg) in the same session
from scipy import ndimage
from scipy.stats import skew, kurtosis
from sklearn.linear_model import LogisticRegressionCV
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RepeatedStratifiedKFold
from concurrent.futures import ThreadPoolExecutor
RN = list(apriori); masks = {n: (rid == names.index(n) + 1) for n in RN}; ST6 = ndimage.generate_binary_structure(3, 1)
def box(m): ix = np.argwhere(m); return tuple(slice(max(a - 2, 0), b + 2) for a, b in zip(ix.min(0), ix.max(0) + 1))
BB = {n: box(masks[n]) for n in RN}
def load_t88_full(sid):
    f = glob.glob(f"{sub_dirs[sid]}/PROCESSED/MPRAGE/T88_111/*_t88_masked_gfc.img")[0]
    return np.asarray(nib.load(f).dataobj).squeeze().astype(np.float32)
def rad_row(sid):
    a = load_t88_full(sid); seg = load_seg(sid)
    if a.shape != rid.shape: raise ValueError(f"{sid}: image grid {a.shape} != ROI grid {rid.shape}")
    m = a > 0; z = (a - a[m].mean()) / (a[m].std() + 1e-6); row = {"subject_id": sid}
    for n in RN:
        sl = BB[n]; r = masks[n][sl] & m[sl]; zz = z[sl]; v = zz[r]; csf_d = ndimage.binary_dilation(seg[sl] == 1, ST6); gm = (seg[sl] == 2) & r
        gmag = np.sqrt(sum(g ** 2 for g in np.gradient(zz)))
        p = np.percentile(v, [10, 25, 50, 75, 90]) if v.size > 20 else [np.nan] * 5; h = np.histogram(np.clip(v, -3, 3), bins=16, range=(-3, 3))[0] / max(v.size, 1)
        for k, val in zip(["mean", "std", "skew", "kurt", "p10", "p25", "p50", "p75", "p90", "entropy", "grad", "gm_csf_contact"],
                          [v.mean(), v.std(), skew(v), kurtosis(v), *p, -(h[h > 0] * np.log(h[h > 0])).sum(), gmag[r].mean(), (gm & csf_d).sum() / max(gm.sum(), 1)]):
            row[f"{n}__r_{k}"] = val
    return row
t0 = time.time()
with ThreadPoolExecutor(4) as ex: rad = pd.DataFrame(list(ex.map(rad_row, SIDS)))
print(f"radiomics table {rad.shape} in {time.time()-t0:.0f}s | NaN cells: {int(rad.isna().sum().sum())}"); rad.to_csv(f"{OUT}/roi_radiomics.csv", index=False); up(f"{OUT}/roi_radiomics.csv", "roi")

# ---- feature sets ----
FR = D[["subject_id"]].merge(roi, on="subject_id", how="left").merge(rad, on="subject_id", how="left").reset_index(drop=True)
def vol(rois): return [c for c in roi.columns if c != "subject_id" and c.split("__")[0] in rois]
def rd(rois): return [c for c in rad.columns if c != "subject_id" and c.split("__")[0] in rois]
HA = ["Hippocampus", "Amygdala"]
SPEC3 = {"VOL hippocampus+amygdala (post-hoc, Cell 17)": vol(HA), "RAD hippocampus+amygdala": rd(HA), "VOL+RAD hippocampus+amygdala": vol(HA) + rd(HA),
         "RAD all 6 ROIs": rd(RN), "VOL+RAD all 6 ROIs (PRIMARY)": vol(RN) + rd(RN)}
def fit_pred3(cols, tr, te):
    m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                      LogisticRegressionCV(Cs=np.logspace(-3, 0, 7), cv=5, scoring="roc_auc", class_weight="balanced", max_iter=5000)).fit(FR.loc[tr, cols], y[tr])
    return m.predict_proba(FR.loc[te, cols])[:, 1]
MARGIN = 0.05
def deltas3(SC, refs):
    out = []
    for ref in refs:
        for k in SC:
            if k == ref: continue
            v = np.array([roc_auc_score(y[i], SC[k][i]) - roc_auc_score(y[i], SC[ref][i]) for i in BI]); lo5, hi5 = np.percentile(v, [5, 95])
            out.append(dict(reference=ref, model=k, delta_auroc=roc_auc_score(y, SC[k]) - roc_auc_score(y, SC[ref]), ci_lo=np.percentile(v, 2.5), ci_hi=np.percentile(v, 97.5),
                            excludes_0=bool(np.percentile(v, 2.5) > 0 or np.percentile(v, 97.5) < 0), equivalent_90ci=bool(lo5 > -MARGIN and hi5 < MARGIN)))
    return pd.DataFrame(out)

# ---- A. identical folds ----
S3 = {"nWBV only": -D.nwbv.values}
for k, cols in SPEC3.items():
    s = np.zeros(N)
    for f in range(5): te = (D.fold == f).values; s[te] = fit_pred3(cols, ~te, te)
    S3[k] = s
rows = []
for k, s in S3.items():
    v = [roc_auc_score(y[i], s[i]) for i in BI]; rows.append(dict(model=k, n_features=len(SPEC3.get(k, [0])), auroc=roc_auc_score(y, s), ci_lo=np.percentile(v, 2.5), ci_hi=np.percentile(v, 97.5),
                                                                   avg_precision=average_precision_score(y, s), sens_at_80spec=sens_at_spec(s)))
TA3 = pd.DataFrame(rows); print("\nA. Identical folds, N =", N); print(TA3.round(3).to_string(index=False))
DA3 = deltas3(S3, ["nWBV only", "VOL hippocampus+amygdala (post-hoc, Cell 17)"]); print("\nPaired delta AUROC:"); print(DA3.round(3).to_string(index=False))

# ---- B. repeated 5x5 CV ----
NREP = 5; rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=NREP, random_state=7); RS3 = {k: np.zeros((NREP, N)) for k in SPEC3}
for j, (tr_i, te_i) in enumerate(rskf.split(np.zeros(N), y)):
    r = j // 5; tr = np.zeros(N, bool); tr[tr_i] = True; te = ~tr
    for k, cols in SPEC3.items(): RS3[k][r, te] = fit_pred3(cols, tr, te)
rows = [dict(model="nWBV only", auroc_mean=roc_auc_score(y, -D.nwbv.values), auroc_sd=0.0)]
for k, M in RS3.items(): a = [roc_auc_score(y, M[r]) for r in range(NREP)]; rows.append(dict(model=k, auroc_mean=np.mean(a), auroc_sd=np.std(a)))
TB3 = pd.DataFrame(rows); print(f"\nB. Repeated 5-fold x {NREP}"); print(TB3.round(3).to_string(index=False))
SCr = {"nWBV only": -D.nwbv.values, **{k: M.mean(0) for k, M in RS3.items()}}; DB3 = deltas3(SCr, ["nWBV only", "VOL hippocampus+amygdala (post-hoc, Cell 17)"])
print("\nPaired delta AUROC on repeat-averaged scores:"); print(DB3.round(3).to_string(index=False))

# ---- C. CDR 0.5 vs 0 and age-matched subset (identical-fold scores) ----
age = D.age.values
def age_match(caliper):
    pos = np.where(y == 1)[0]; neg = list(np.where(y == 0)[0]); keep = []
    for i in pos[np.argsort(age[pos])]:
        if not neg: break
        d = np.abs(age[neg] - age[i]); j = int(np.argmin(d))
        if d[j] <= caliper: keep += [i, neg.pop(j)]
    return np.array(sorted(keep))
rng = np.random.default_rng(11); rows = []; subsets = {"CDR0.5_vs_0": np.where(D.cdr.isin([0, 0.5]).values)[0], "age-matched(2y)": age_match(2.0)}
for sn, idx in subsets.items():
    ys = y[idx]; bs = [b for b in (rng.integers(0, len(idx), len(idx)) for _ in range(1000)) if len(np.unique(ys[b])) == 2]
    for k, s in S3.items():
        v = [roc_auc_score(ys[b], s[idx][b]) for b in bs]; rows.append(dict(subset=sn, n=len(idx), model=k, auroc=roc_auc_score(ys, s[idx]), ci_lo=np.percentile(v, 2.5), ci_hi=np.percentile(v, 97.5)))
TC3 = pd.DataFrame(rows); print("\nC. Subsets"); print(TC3.round(3).to_string(index=False))

P = "VOL+RAD all 6 ROIs (PRIMARY)"; pr = DA3[(DA3.reference == "nWBV only") & (DA3.model == P)].iloc[0]; pr2 = DB3[(DB3.reference == "nWBV only") & (DB3.model == P)].iloc[0]
print(f"\nPRIMARY vs nWBV: identical folds {pr.delta_auroc:+.3f} (95% CI {pr.ci_lo:+.3f} to {pr.ci_hi:+.3f}) | repeated CV {pr2.delta_auroc:+.3f} (95% CI {pr2.ci_lo:+.3f} to {pr2.ci_hi:+.3f})")
print("Models tried in this cell: 5 (report all). Success criterion fixed in advance: both CIs exclude 0.")
for nm, t in [("rad_identical_folds", TA3), ("rad_paired_deltas", DA3), ("rad_repeatedcv", TB3), ("rad_repeatedcv_deltas", DB3), ("rad_subsets", TC3)]: t.to_csv(f"{OUT}/{nm}.csv", index=False); up(f"{OUT}/{nm}.csv", "roi")
pd.DataFrame(S3).assign(subject_id=D.subject_id, fold=D.fold, cdr_binary=y).to_csv(f"{OUT}/rad_oof_scores.csv", index=False); up(f"{OUT}/rad_oof_scores.csv", "roi")
