# ===== CELL 3 — Cross-fitted brain-age models (normal-only training, bias-corrected) =====
from sklearn.model_selection import RepeatedStratifiedKFold, KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.svm import SVR
from sklearn.metrics import roc_auc_score, mean_absolute_error
from scipy.stats import spearmanr

LABEL_MAP = {1: "csf", 2: "gm", 3: "wm"}     # verify against Cell 2 sanity table; edit if the labels differ
df = pd.read_csv(f"{OUT}/features.csv")
for k, n in LABEL_MAP.items(): df[f"{n}_vol"] = df[f"vol_{k}"]
df["tissue_total"] = df[["csf_vol", "gm_vol", "wm_vol"]].sum(axis=1)
for n in ["csf", "gm", "wm"]: df[f"{n}_frac"] = df[f"{n}_vol"] / df["tissue_total"]
df["sex_f"] = (df.sex == "F").astype(int)
FEATS = ["nwbv", "etiv", "asf", "csf_frac", "gm_frac", "wm_frac", "gm_vol", "wm_vol"]
df = df.dropna(subset=FEATS + ["age"]).reset_index(drop=True)
lab = df[df.cdr.notna()].reset_index(drop=True)
unl = df[df.cdr.isna() & (df.age < 60)].reset_index(drop=True)   # presumed cognitively normal (CDR not recorded <60y)
print(f"labeled {len(lab)} (CDR0 {int((lab.cdr==0).sum())}) | unlabeled <60y presumed normal {len(unl)} | unlabeled >=60y excluded {int((df.cdr.isna() & (df.age>=60)).sum())}")

MODELS = {"ridge": lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20))),
          "svr":   lambda: make_pipeline(StandardScaler(), SVR(C=10, epsilon=1.0))}
VARIANTS = ["CDR0", "CDR0+young"]   # which subjects train the 'normal' age model
N_REP = 5

def crossfit(variant, mk, seed=0):
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=N_REP, random_state=seed); rec = []
    for sp, (tr, te) in enumerate(rskf.split(lab, lab.cdr_binary)):
        trn = lab.iloc[tr]; trn = trn[trn.cdr == 0]
        if variant == "CDR0+young": trn = pd.concat([trn, unl], ignore_index=True)
        oof = cross_val_predict(mk(), trn[FEATS], trn.age, cv=KFold(5, shuffle=True, random_state=sp))
        a, b = np.polyfit(trn.age.values, oof, 1)                  # bias line fitted on training normals only
        if a < 0.2: a, b = 1.0, 0.0
        m = mk().fit(trn[FEATS], trn.age); p = m.predict(lab.iloc[te][FEATS]); pc = (p - b) / a
        rec += [(lab.subject_id[i], sp // 5, x, y) for i, x, y in zip(te, p, pc)]
    r = pd.DataFrame(rec, columns=["subject_id", "rep", "pred", "pred_corr"])
    g = r.groupby("subject_id")[["pred", "pred_corr"]].mean().reset_index()   # mean over repeats; every prediction is from a model that never saw the subject
    out = lab.merge(g, on="subject_id"); out["bag_raw"] = out.pred - out.age; out["bag_corr"] = out.pred_corr - out.age
    return out

BAG = {}
for v in VARIANTS:
    for mn, mk in MODELS.items():
        BAG[(v, mn)] = crossfit(v, mk); b = BAG[(v, mn)]; n0 = b[b.cdr == 0]
        print(f"{v:11s} {mn:5s} | MAE on held-out CDR0: {mean_absolute_error(n0.age, n0.pred_corr):.2f}y | mean BAG CDR0 {n0.bag_corr.mean():+.2f}  CDR>0 {b[b.cdr>0].bag_corr.mean():+.2f}")
        b.to_csv(f"{OUT}/bag_{v}_{mn}.csv", index=False)
for k in os.listdir(OUT):
    if k.startswith("bag_"): up(f"{OUT}/{k}", "brainage")
