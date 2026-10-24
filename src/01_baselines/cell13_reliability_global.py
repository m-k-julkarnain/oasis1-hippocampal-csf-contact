# ===== CELL 13 — Test-retest reliability on the 20 repeat sessions (MR1 vs MR2, <=90 days apart) =====
# needs Cell 1, Cell 2 (one(), ROOT), Cell 3 (lab, unl, FEATS, imports), Cell 8, Cell 9, Cell 10 checkpoints, and blocks.npz from Cell 5
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
mr2 = sorted(glob.glob(f"{ROOT}/OAS1_*_MR2")); print("MR2 sessions found:", len(mr2))
for d in mr2: sub_dirs[os.path.basename(d)] = d
pairs = [(os.path.basename(d)[:-1] + "1", os.path.basename(d)) for d in mr2 if os.path.basename(d)[:-1] + "1" in SIDX]

# ---- per-session features (same code path for both sessions)
def sess_row(sid):
    r = one(sub_dirs[sid]); r["session"] = sid
    for k, n in LABEL_MAP.items(): r[f"{n}_vol"] = r.get(f"vol_{k}", np.nan)
    tot = r["csf_vol"] + r["gm_vol"] + r["wm_vol"]; r["csf_frac"] = r["csf_vol"] / tot; r["gm_frac"] = r["gm_vol"] / tot; r["wm_frac"] = r["wm_vol"] / tot
    r["nwbv_seg"] = (r["gm_vol"] + r["wm_vol"]) / tot; return r
ok = []
for a, b in pairs:
    try: ok.append((a, b, sess_row(a), sess_row(b), load_t88(b)))
    except Exception as e: print("skip", b, e)
print("usable pairs:", len(ok)); R1 = pd.DataFrame([o[2] for o in ok]); R2s = pd.DataFrame([o[3] for o in ok])
print("age range", R1.age.min(), "-", R1.age.max(), "| CDR values (MR1):", R1.cdr.value_counts(dropna=False).to_dict())

# ---- block-feature brain-age (ridge trained on presumed normals, excluding these subjects)
BLK = 8; z = np.load(f"{OUT}/blocks.npz", allow_pickle=True); XB = z["X"].astype(np.float32); SIDm = {s: i for i, s in enumerate(z["sids"])}; keep = z["keep"]
def blocks_vec(sid):
    f = glob.glob(f"{sub_dirs[sid]}/FSL_SEG/*_fseg.img")[0]; a = np.asarray(nib.load(f).dataobj).squeeze().astype(np.uint8)
    nx, ny, nz = [s // BLK for s in a.shape]; a = a[:nx * BLK, :ny * BLK, :nz * BLK].reshape(nx, BLK, ny, BLK, nz, BLK)
    return np.stack([(a == k).mean(axis=(1, 3, 5)) for k in (1, 2, 3)], -1).astype(np.float32)[keep, :].reshape(-1)
excl = {o[0] for o in ok}
nm = pd.concat([lab[(lab.cdr == 0) & ~lab.subject_id.isin(excl)][["subject_id", "age"]], unl[~unl.subject_id.isin(excl)][["subject_id", "age"]]], ignore_index=True)
rm = RidgeCV(alphas=np.logspace(0, 5, 11)).fit(XB[[SIDm[s] for s in nm.subject_id]], nm.age.values)
bag_blk1 = rm.predict(np.stack([XB[SIDm[o[0]]] for o in ok])) - R1.age.values
bag_blk2 = rm.predict(np.stack([blocks_vec(o[1]) for o in ok])) - R2s.age.values

# ---- CNN ensemble of the 5 fold models (reliability does not depend on which subjects a model saw)
def load_ck(tag):
    ck = torch.load(f"{CNN_DIR}/{tag}.pt", map_location=DEVICE); m = Net3D().to(DEVICE); m.load_state_dict(ck["state"]); m.eval(); return m, ck["mu"], ck["sd"]
@torch.no_grad()
def pred_t(m, X):
    out = []
    for k in range(0, len(X), 8):
        with torch.autocast(DEVICE, dtype=torch.float16, enabled=DEVICE == "cuda"): o, _ = m(X[k:k + 8].float().to(DEVICE))
        out.append(o.float().cpu().numpy())
    return np.concatenate(out)
X1 = IMG[[SIDX[o[0]] for o in ok]]; X2 = torch.stack([o[4] for o in ok]).to(DEVICE)
age1, age2, ps1, ps2, pp1, pp2 = [], [], [], [], [], []
for f in range(5):
    m, mu, sd = load_ck(f"age_f{f}"); age1.append(pred_t(m, X1) * sd + mu); age2.append(pred_t(m, X2) * sd + mu)
    m, _, _ = load_ck(f"cdr_scratch_f{f}"); ps1.append(1 / (1 + np.exp(-pred_t(m, X1)))); ps2.append(1 / (1 + np.exp(-pred_t(m, X2))))
    m, _, _ = load_ck(f"cdr_agepre_f{f}"); pp1.append(1 / (1 + np.exp(-pred_t(m, X1)))); pp2.append(1 / (1 + np.exp(-pred_t(m, X2))))
bag_cnn1 = np.mean(age1, 0) - R1.age.values; bag_cnn2 = np.mean(age2, 0) - R2s.age.values

M = {"nWBV (reported)": (R1.nwbv.values, R2s.nwbv.values), "nWBV (from segmentation)": (R1.nwbv_seg.values, R2s.nwbv_seg.values),
     "CSF fraction": (R1.csf_frac.values, R2s.csf_frac.values), "Block-ridge brain-age gap (y)": (bag_blk1, bag_blk2),
     "CNN brain-age gap (y)": (bag_cnn1, bag_cnn2), "CNN CDR prob, scratch": (np.mean(ps1, 0), np.mean(ps2, 0)),
     "CNN CDR prob, age-pretrained": (np.mean(pp1, 0), np.mean(pp2, 0))}
def icc21(a, b):
    X = np.column_stack([a, b]); n, k = X.shape; gm = X.mean(); msr = k * ((X.mean(1) - gm) ** 2).sum() / (n - 1); msc = n * ((X.mean(0) - gm) ** 2).sum() / (k - 1)
    mse = ((X - X.mean(1, keepdims=True) - X.mean(0, keepdims=True) + gm) ** 2).sum() / ((n - 1) * (k - 1)); return (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)
rng = np.random.default_rng(0); n = len(ok); BS = [rng.integers(0, n, n) for _ in range(2000)]; rows = []
for k, (a, b) in M.items():
    dlt = b - a; wsd = np.sqrt(np.mean(dlt ** 2) / 2); bsd = np.concatenate([a, b]).std(ddof=1); ic = [icc21(a[i], b[i]) for i in BS if np.ptp(a[i]) > 0]
    flips = int(((a >= 0.5) != (b >= 0.5)).sum()) if "prob" in k else np.nan
    rows.append(dict(measure=k, n=n, icc21=icc21(a, b), icc_ci_lo=np.percentile(ic, 2.5), icc_ci_hi=np.percentile(ic, 97.5), mean_diff=dlt.mean(), sd_diff=dlt.std(ddof=1),
                     within_subj_sd=wsd, between_subj_sd=bsd, wsd_over_bsd=wsd / bsd, repeatability_coef=2.77 * wsd, class_flips_at_0p5=flips))
RT = pd.DataFrame(rows); print(RT.round(3).to_string(index=False)); RT.to_csv(f"{OUT}/test_retest.csv", index=False)
pd.DataFrame({f"{k}|MR1": v[0] for k, v in M.items()} | {f"{k}|MR2": v[1] for k, v in M.items()} | {"subject": [o[0] for o in ok]}).to_csv(f"{OUT}/test_retest_values.csv", index=False)
fig, ax = plt.subplots(2, 4, figsize=(15, 7)); ax = ax.ravel()
for i, (k, (a, b)) in enumerate(M.items()):
    ax[i].scatter(a, b, s=18); lo, hi = min(a.min(), b.min()), max(a.max(), b.max()); ax[i].plot([lo, hi], [lo, hi], "k--", lw=.7); ax[i].set_title(f"{k}\nICC {RT.icc21[i]:.2f}", fontsize=8)
    ax[i].set_xlabel("MR1"); ax[i].set_ylabel("MR2")
ax[-1].axis("off"); plt.tight_layout(); plt.savefig(f"{OUT}/test_retest.png", dpi=110); plt.close()
for k in ["test_retest.csv", "test_retest_values.csv", "test_retest.png"]: up(f"{OUT}/{k}", "reliability")
