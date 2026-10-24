# ===== CELL 6 — Brain-age from block features (cross-fitted, normal-only training, bias-corrected) =====
# needs Cell 1 and Cell 3 (uses lab, unl, imports) in the same session
z = np.load(f"{OUT}/blocks.npz", allow_pickle=True); XB = z["X"].astype(np.float32); SID = {s: i for i, s in enumerate(z["sids"])}
MODELS_BLK = {"blk_ridge": lambda: RidgeCV(alphas=np.logspace(0, 5, 11))}
N_REP_B = 3

def crossfit_blk(variant, mk, seed=0):
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=N_REP_B, random_state=seed); rec = []
    ix = lambda d: [SID[s] for s in d.subject_id]
    for sp, (tr, te) in enumerate(rskf.split(lab, lab.cdr_binary)):
        trn = lab.iloc[tr]; trn = trn[trn.cdr == 0]
        if variant == "CDR0+young": trn = pd.concat([trn, unl], ignore_index=True)
        Xtr, ytr = XB[ix(trn)], trn.age.values
        oof = cross_val_predict(mk(), Xtr, ytr, cv=KFold(5, shuffle=True, random_state=sp))
        a, b = np.polyfit(ytr, oof, 1)
        if a < 0.2: a, b = 1.0, 0.0
        m = mk().fit(Xtr, ytr); ted = lab.iloc[te]; p = m.predict(XB[ix(ted)]); pc = (p - b) / a
        rec += [(s, sp // 5, x, y) for s, x, y in zip(ted.subject_id, p, pc)]
    r = pd.DataFrame(rec, columns=["subject_id", "rep", "pred", "pred_corr"])
    g = r.groupby("subject_id")[["pred", "pred_corr"]].mean().reset_index()
    out = lab.merge(g, on="subject_id"); out["bag_raw"] = out.pred - out.age; out["bag_corr"] = out.pred_corr - out.age
    return out

BAG_BLK = {}
for v in VARIANTS:
    for mn, mk in MODELS_BLK.items():
        BAG_BLK[(v, mn)] = crossfit_blk(v, mk); b = BAG_BLK[(v, mn)]; n0 = b[b.cdr == 0]
        print(f"{v:11s} {mn} | held-out CDR0 MAE raw {mean_absolute_error(n0.age, n0.pred):.2f}y, corrected {mean_absolute_error(n0.age, n0.pred_corr):.2f}y"
              f" | mean BAG CDR0 {n0.bag_corr.mean():+.2f}  CDR>0 {b[b.cdr>0].bag_corr.mean():+.2f}")
        b.to_csv(f"{OUT}/bag_{v}_{mn}.csv", index=False); up(f"{OUT}/bag_{v}_{mn}.csv", "brainage")
