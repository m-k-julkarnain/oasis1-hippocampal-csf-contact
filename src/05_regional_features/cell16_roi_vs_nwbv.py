# ===== CELL 16 — Do regional (ROI) features beat nWBV?  Identical folds + repeated CV + stage-stratified + equivalence margin =====
# needs Cell 12 in the same session (D, S, y, N, BI, base, XD, sens_at_spec, NAMES) and roi_features.csv / roi_sets.json from Cell 15
from sklearn.model_selection import RepeatedStratifiedKFold
MARGIN = 0.05                                                                      # pre-specified equivalence margin (AUROC)
roi = pd.read_csv(f"{OUT}/roi_features.csv"); sets = json.load(open(f"{OUT}/roi_sets.json"))
R = D[["subject_id"]].merge(roi, on="subject_id", how="left"); assert R.drop(columns="subject_id").notna().all().all(), "ROI features missing for some subjects"
ALL = [c for c in roi.columns if c != "subject_id"]; MTL = [c for c in ALL if c.split("__")[0] in sets["apriori"]]
XR = pd.concat([D[base].reset_index(drop=True), R[ALL].reset_index(drop=True)], axis=1); print(f"ROI features: all {len(ALL)} | a-priori MTL+ventricle {len(MTL)}")
SPEC = {"ROI MTL+vent (MRI-only)": (MTL, 0.1), "ROI MTL+vent + nWBV": (MTL + ["nwbv"], 0.1), "ROI MTL+vent + age,sex,eTIV,nWBV": (MTL + base, 0.1),
        "ROI all (MRI-only)": (ALL, 0.02), "ROI all + age,sex,eTIV,nWBV": (ALL + base, 0.02)}
def fit_pred(cols, C, tr, te):
    m = make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=5000)).fit(XR.loc[tr, cols], y[tr])
    return m.predict_proba(XR.loc[te, cols])[:, 1]
def blocks_pred(tr, te):
    m = make_pipeline(StandardScaler(), PCA(n_components=40, random_state=0), LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000)).fit(XD[tr], y[tr])
    return m.predict_proba(XD[te])[:, 1]

# ---- A. identical folds (same 5 folds as the CNN) ----
S2 = {k: S[k] for k in ["nWBV only", "LR age+sex+eTIV+nWBV", "Blocks MRI-only (PCA+LR)", "CNN age-pretrained"]}
for k, (cols, C) in SPEC.items():
    s = np.zeros(N)
    for f in range(5): te = (D.fold == f).values; s[te] = fit_pred(cols, C, ~te, te)
    S2[k] = s
rows = []
for k, s in S2.items():
    v = [roc_auc_score(y[i], s[i]) for i in BI]
    rows.append(dict(model=k, auroc=roc_auc_score(y, s), ci_lo=np.percentile(v, 2.5), ci_hi=np.percentile(v, 97.5), avg_precision=average_precision_score(y, s),
                     sens_at_80spec=sens_at_spec(s), brier=brier_score_loss(y, s) if k != "nWBV only" else np.nan))
TA = pd.DataFrame(rows); print("\nA. Identical folds, N =", N); print(TA.round(3).to_string(index=False))

def deltas(SC, refs):
    out = []
    for ref in refs:
        for k in SC:
            if k == ref: continue
            v = np.array([roc_auc_score(y[i], SC[k][i]) - roc_auc_score(y[i], SC[ref][i]) for i in BI]); lo5, hi5 = np.percentile(v, [5, 95])
            out.append(dict(reference=ref, model=k, delta_auroc=roc_auc_score(y, SC[k]) - roc_auc_score(y, SC[ref]), ci_lo=np.percentile(v, 2.5), ci_hi=np.percentile(v, 97.5),
                            equivalent_90ci=bool(lo5 > -MARGIN and hi5 < MARGIN), p_within_margin=float(np.mean(np.abs(v) < MARGIN))))
    return pd.DataFrame(out)
DA = deltas(S2, ["nWBV only", "LR age+sex+eTIV+nWBV"]); DA["excludes_0"] = (DA.ci_lo > 0) | (DA.ci_hi < 0)
print(f"\nPaired delta AUROC (equivalence margin +/-{MARGIN}):"); print(DA.round(3).to_string(index=False))

# ---- B. stage-stratified AUROC + MMSE association (out-of-fold scores) ----
rng = np.random.default_rng(3); rows = []
for sn, msk in {"CDR0.5_vs_0": D.cdr.isin([0, 0.5]).values, "CDR>=1_vs_0": D.cdr.isin([0, 1, 2]).values}.items():
    idx = np.where(msk)[0]; ys = y[idx]; bs = [b for b in (rng.integers(0, len(idx), len(idx)) for _ in range(1000)) if len(np.unique(ys[b])) == 2]
    for k, s in S2.items():
        v = [roc_auc_score(ys[b], s[idx][b]) for b in bs]
        rows.append(dict(task=sn, model=k, n=len(idx), auroc=roc_auc_score(ys, s[idx]), ci_lo=np.percentile(v, 2.5), ci_hi=np.percentile(v, 97.5)))
TB = pd.DataFrame(rows); print("\nB. Stage-stratified AUROC"); print(TB.round(3).to_string(index=False))
mm = D.mmse.notna().values; TM = pd.DataFrame([dict(model=k, spearman_mmse=spearmanr(s[mm], D.mmse[mm])[0]) for k, s in S2.items()])
print("\nOut-of-fold score vs MMSE (negative = higher score, lower MMSE):"); print(TM.round(3).to_string(index=False))

# ---- C. repeated stratified 5x5 CV for the cheap models (own folds; tightens the estimate) ----
NREP = 5; rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=NREP, random_state=7); RS = {k: np.zeros((NREP, N)) for k in ["LR age+sex+eTIV+nWBV", "Blocks MRI-only (PCA+LR)"] + list(SPEC)}
for j, (tr_i, te_i) in enumerate(rskf.split(np.zeros(N), y)):
    r = j // 5; tr = np.zeros(N, bool); tr[tr_i] = True; te = ~tr
    RS["LR age+sex+eTIV+nWBV"][r, te] = fit_pred(base, 1.0, tr, te); RS["Blocks MRI-only (PCA+LR)"][r, te] = blocks_pred(tr, te)
    for k, (cols, C) in SPEC.items(): RS[k][r, te] = fit_pred(cols, C, tr, te)
rows = [dict(model="nWBV only", auroc_mean=roc_auc_score(y, -D.nwbv.values), auroc_sd=0.0)]
for k, M in RS.items(): a = [roc_auc_score(y, M[r]) for r in range(NREP)]; rows.append(dict(model=k, auroc_mean=np.mean(a), auroc_sd=np.std(a)))
TC = pd.DataFrame(rows); print(f"\nC. Repeated 5-fold x {NREP}: pooled out-of-fold AUROC per repeat"); print(TC.round(3).to_string(index=False))
SC = {"nWBV only": -D.nwbv.values, **{k: M.mean(0) for k, M in RS.items()}}
DC = deltas(SC, ["nWBV only", "LR age+sex+eTIV+nWBV"]); DC["excludes_0"] = (DC.ci_lo > 0) | (DC.ci_hi < 0)
print("\nPaired delta AUROC on repeat-averaged scores:"); print(DC.round(3).to_string(index=False))

# ---- save + figure ----
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(7.5, 4.8)); yy = np.arange(len(TA))[::-1]
ax.errorbar(TA.auroc, yy, xerr=[TA.auroc - TA.ci_lo, TA.ci_hi - TA.auroc], fmt="o", capsize=3); ax.set_yticks(yy); ax.set_yticklabels(TA.model, fontsize=8)
ax.axvline(0.5, c="gray", lw=.5); ax.axvline(TA.auroc[TA.model == "nWBV only"].iloc[0], c="C3", lw=.8, ls="--"); ax.set_xlabel("AUROC (95% subject bootstrap CI)"); ax.set_title("Regional ROI features vs nWBV (identical folds)")
plt.tight_layout(); plt.savefig(f"{OUT}/roi_benchmark.png", dpi=120); plt.close()
for nm, t in [("roi_benchmark", TA), ("roi_paired_deltas", DA), ("roi_stage", TB), ("roi_mmse", TM), ("roi_repeatedcv", TC), ("roi_repeatedcv_deltas", DC)]: t.to_csv(f"{OUT}/{nm}.csv", index=False)
pd.DataFrame(S2).assign(subject_id=D.subject_id, fold=D.fold, cdr_binary=y).to_csv(f"{OUT}/roi_oof_scores.csv", index=False)
for k in ["roi_benchmark.csv", "roi_paired_deltas.csv", "roi_stage.csv", "roi_mmse.csv", "roi_repeatedcv.csv", "roi_repeatedcv_deltas.csv", "roi_oof_scores.csv", "roi_benchmark.png"]: up(f"{OUT}/{k}", "roi")
