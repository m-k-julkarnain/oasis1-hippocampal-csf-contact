# ===== CELL 15 — Regional (ROI) tissue features: Harvard-Oxford ROIs mapped onto the atlas-space FSL_SEG =====
# needs Cell 1 and Cell 2 (features.csv). Internet must be ON (pip install + atlas download). No GPU needed.
import itertools
from concurrent.futures import ThreadPoolExecutor
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "nilearn"], check=False)
if "ROOT" not in globals():
    ROOT = pd.Series([os.path.dirname(p) for p in glob.glob(f"{DATA_DIR}/**/OAS1_*_MR1", recursive=True) if os.path.isdir(p)]).value_counts().index[0]
sub_dirs = {os.path.basename(d): d for d in sorted(glob.glob(f"{ROOT}/OAS1_*_MR1"))}
SIDS = list(pd.read_csv(f"{OUT}/features.csv").subject_id)
def load_seg(sid):
    f = glob.glob(f"{sub_dirs[sid]}/FSL_SEG/*_fseg.img")[0]
    return np.asarray(nib.load(f).dataobj).squeeze().astype(np.uint8)       # 0 bg, 1 CSF, 2 GM, 3 WM (atlas space)

# --- fetch atlases
from nilearn import datasets
from scipy.ndimage import binary_closing, binary_fill_holes
def arr(x): x = nib.load(x) if isinstance(x, str) else x; return np.asarray(x.dataobj).squeeze()
ho_c = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-1mm"); ho_s = datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr25-1mm")
cort_arr, cort_labels = arr(ho_c.maps).astype(np.int32), list(ho_c.labels)
sub_arr, sub_labels = arr(ho_s.maps).astype(np.int32), list(ho_s.labels)
# --- end fetch
assert cort_arr.shape == sub_arr.shape, (cort_arr.shape, sub_arr.shape)
mni_mask = binary_fill_holes(binary_closing((cort_arr > 0) | (sub_arr > 0), iterations=3))   # brain extent taken from the atlases themselves (same grid)

# ---- 1. align the MNI/Harvard-Oxford grid to the OASIS atlas grid (axis permutation/flips + per-axis bounding-box scaling), checked by Dice overlap
ref = np.mean([load_seg(s) > 0 for s in SIDS[::10]], 0) > 0.5; G = ref.shape
def crop_sl(m): ix = np.argwhere(m); return tuple(slice(a, b) for a, b in zip(ix.min(0), ix.max(0) + 1))
def nn_idx(src, dst): return np.ix_(*[np.minimum(((np.arange(d) + .5) * s / d).astype(int), s - 1) for s, d in zip(src, dst)])
def orient(a, perm, flips):
    a = np.transpose(a, perm)
    for ax, f in enumerate(flips):
        if f: a = np.flip(a, ax)
    return a
rs = crop_sl(ref); refc = ref[rs]; cands = []
for perm in itertools.permutations(range(3)):
    for flips in itertools.product([0, 1], repeat=3):
        m = orient(mni_mask, perm, flips); mc = m[crop_sl(m)]; mr = mc[nn_idx(mc.shape, refc.shape)]
        cands.append((2 * (mr & refc).sum() / (mr.sum() + refc.sum()), perm, flips))
cands.sort(key=lambda t: -t[0]); print("top orientations (Dice, perm, flips):")
for c in cands[:4]: print(f"  {c[0]:.3f} {c[1]} {c[2]}")
dice, perm, flips = cands[0]
if dice < 0.80: raise RuntimeError(f"atlas alignment poor (Dice {dice:.2f}); paste the printout above to Claude")
m0 = orient(mni_mask, perm, flips); cs0 = crop_sl(m0); IDX = nn_idx(m0[cs0].shape, refc.shape)
def to_grid(a):
    out = np.zeros(G, np.int32); out[rs] = orient(a, perm, flips)[cs0][IDX]; return out

# ---- 2. ROI map (left+right merged); cortical atlas first, subcortical atlas overrides on overlap
def clean(n): return re.sub(r"[^A-Za-z0-9]+", "_", re.sub(r"^(Left|Right) ", "", n)).strip("_")
rid = np.zeros(G, np.int32); names = []
for arr_, labs, skip in [(to_grid(cort_arr), cort_labels, []), (to_grid(sub_arr), sub_labels, ["Cerebral_White_Matter", "Cerebral_Cortex"])]:
    for i, n in enumerate(labs):
        if i == 0 or n.lower() == "background": continue
        c = clean(n)
        if c in skip: continue
        if c not in names: names.append(c)
        rid[arr_ == i] = names.index(c) + 1
nR = len(names) + 1; rid_f = rid.ravel().astype(np.int64) * 4
sizes = np.bincount(rid.ravel(), minlength=nR)[1:]; print(f"{len(names)} bilateral ROIs; voxel counts min/median/max: {sizes.min()}/{int(np.median(sizes))}/{sizes.max()}")

# ---- 3. per-subject tissue content inside each ROI
def roi_row(sid):
    a = load_seg(sid).ravel()
    if a.size != rid_f.size: raise ValueError(f"{sid}: seg shape differs from reference grid")
    c = np.bincount(rid_f + np.minimum(a, 3), minlength=nR * 4).reshape(nR, 4)[1:]       # cols: bg, CSF, GM, WM
    tot = np.maximum(c[:, 1:].sum(1), 1); row = {"subject_id": sid}
    for n, (csf, gm), t in zip(names, c[:, 1:3], tot):
        row[f"{n}__csf"] = csf; row[f"{n}__gm"] = gm; row[f"{n}__csf_frac"] = csf / t; row[f"{n}__gm_frac"] = gm / t
    return row
t0 = time.time()
with ThreadPoolExecutor(4) as ex: roi = pd.DataFrame(list(ex.map(roi_row, SIDS)))
roi = roi.loc[:, ~(roi.columns.str.contains("__") & (roi.drop(columns="subject_id").std() == 0).reindex(roi.columns).fillna(False))]
print(f"ROI table {roi.shape} in {time.time()-t0:.0f}s")

# ---- 4. alignment sanity: hippocampus & thalamus should be mostly GM, lateral ventricle mostly CSF
for key, col in [("Hippocamp", "gm_frac"), ("Amygdala", "gm_frac"), ("Thalamus", "gm_frac"), ("Lateral_Ventricle", "csf_frac")]:
    for n in [n for n in names if key in n]: print(f"  SANITY {n}: mean {col} = {roi[f'{n}__{col}'].mean():.2f}  (expect > ~0.5)")
apriori = [n for n in names if re.search(r"Hippocamp|Amygdala|Parahippocampal|Temporal_Pole|Lateral_Ventricle", n)]; print("a-priori ROIs:", apriori)
roi.to_csv(f"{OUT}/roi_features.csv", index=False); json.dump({"apriori": apriori, "orientation": [float(dice), list(perm), list(flips)]}, open(f"{OUT}/roi_sets.json", "w"))
for k in ["roi_features.csv", "roi_sets.json"]: up(f"{OUT}/{k}", "roi")
