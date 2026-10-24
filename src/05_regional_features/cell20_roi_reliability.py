# ===== CELL 20 — Test-retest reliability of regional (ROI) features on the 20 repeat sessions, anchored to the CDR group difference =====
# needs Cell 1, Cell 3 (imports), Cell 12 (D, y), Cell 15 (roi, roi_row, load_seg, sub_dirs, ROOT, SIDS, apriori) in the same session. Cell 19 optional (adds radiomics contact feature).
mr2 = sorted(glob.glob(f"{ROOT}/OAS1_*_MR2")); print("MR2 sessions found:", len(mr2))
for d in mr2: sub_dirs[os.path.basename(d)] = d
pairs = [(os.path.basename(d)[:-1] + "1", os.path.basename(d)) for d in mr2 if os.path.basename(d)[:-1] + "1" in set(SIDS)]
def nwbv_seg(sid):
    s = load_seg(sid); c, g, w = [(s == k).sum() for k in (1, 2, 3)]; return (g + w) / (c + g + w)
def feats(sid):
    r = roi_row(sid); r["nWBV (from segmentation)"] = nwbv_seg(sid)
    if "rad_row" in globals(): r.update({k: v for k, v in rad_row(sid).items() if k.endswith("gm_csf_contact")})
    return r
F1 = pd.DataFrame([feats(a) for a, b in pairs]); F2 = pd.DataFrame([feats(b) for a, b in pairs]); n = len(pairs); print("repeat pairs:", n)
FC = pd.DataFrame([feats(s) for s in D.subject_id]); lab_y = y                                    # labeled cohort, MR1, for the CDR group difference
keys = ["nWBV (from segmentation)"] + [c for c in F1.columns if c != "subject_id" and c.split("__")[0] in apriori and not c.endswith("__csf") and not c.endswith("__gm")] \
       + [c for c in F1.columns if c.endswith("__gm") and c.split("__")[0] in ("Hippocampus", "Amygdala")]
def icc21(a, b):
    X = np.column_stack([a, b]); k = 2; gm = X.mean(); msr = k * ((X.mean(1) - gm) ** 2).sum() / (n - 1); msc = n * ((X.mean(0) - gm) ** 2).sum() / (k - 1)
    mse = ((X - X.mean(1, keepdims=True) - X.mean(0, keepdims=True) + gm) ** 2).sum() / ((n - 1) * (k - 1)); return (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)
rng = np.random.default_rng(0); BS = [rng.integers(0, n, n) for _ in range(2000)]; rows = []
for k in keys:
    a, b = F1[k].values.astype(float), F2[k].values.astype(float)
    if np.ptp(a) == 0: continue
    wsd = np.sqrt(np.mean((b - a) ** 2) / 2); rc = 2.77 * wsd; ic = [icc21(a[i], b[i]) for i in BS if np.ptp(a[i]) > 0]
    g1, g0 = FC.loc[lab_y == 1, k].astype(float), FC.loc[lab_y == 0, k].astype(float); diff = g1.mean() - g0.mean(); psd = np.sqrt((g1.var() + g0.var()) / 2)
    rows.append(dict(feature=k, icc=icc21(a, b), icc_lo=np.percentile(ic, 2.5), icc_hi=np.percentile(ic, 97.5), within_sd=wsd, repeat_coef=rc,
                     cv_pct=100 * wsd / abs(np.mean(np.r_[a, b])), group_diff_cdr_gt0_minus_0=diff, cohen_d=diff / psd, rc_over_group_diff=rc / abs(diff)))
TR = pd.DataFrame(rows).sort_values("cohen_d", key=lambda s: -s.abs()); print(f"\nTest-retest on {n} repeat pairs: ICC(2,1), repeatability coefficient RC = 2.77 x within-subject SD, and RC relative to the CDR>0 vs CDR=0 group difference")
print(TR.round(4).to_string(index=False))
print("\nFor comparison (Cell 13): block-ridge brain-age gap ICC 0.96, RC 3.4 y; CNN brain-age gap ICC 0.74, RC 2.85 y; nWBV ICC 0.975.")
print("Reading: rc_over_group_diff < 1 means a single re-scan changes the feature by less than the typical CDR>0 vs CDR=0 difference.")
TR.to_csv(f"{OUT}/roi_reliability.csv", index=False); up(f"{OUT}/roi_reliability.csv", "roi")
