# ===== CELL 2 — Build feature table: clinical data + tissue volumes from FSL_SEG (MR1 only) =====
from concurrent.futures import ThreadPoolExecutor
cands = [os.path.dirname(p) for p in glob.glob(f"{DATA_DIR}/**/OAS1_*_MR1", recursive=True) if os.path.isdir(p)]
ROOT = pd.Series(cands).value_counts().index[0]; dirs = sorted(glob.glob(f"{ROOT}/OAS1_*_MR1")); print("root:", ROOT, "| subjects:", len(dirs))

def parse_txt(path):
    d = {}
    if os.path.exists(path):
        for line in open(path, errors="ignore"):
            m = re.match(r"^\s*([A-Za-z/ ]+?)\s*:\s*(.*?)\s*$", line)
            if m: d[m.group(1).strip().lower()] = m.group(2).strip()
    return d
def num(v):
    try: return float(v)
    except Exception: return np.nan

def one(d):
    sid = os.path.basename(d); t = parse_txt(f"{d}/{sid}.txt")
    row = dict(subject_id=sid, cdr=num(t.get("cdr")), mmse=num(t.get("mmse")), age=num(t.get("age")), sex=t.get("m/f"),
               educ=num(t.get("educ")), ses=num(t.get("ses")), etiv=num(t.get("etiv")), asf=num(t.get("asf")), nwbv=num(t.get("nwbv")))
    seg = glob.glob(f"{d}/FSL_SEG/*_fseg.img")
    if seg:
        img = nib.load(seg[0]); a = np.asarray(img.dataobj).squeeze().astype(np.int32)
        vox = float(np.prod(img.header.get_zooms()[:3])); labs, cnt = np.unique(a, return_counts=True)
        for l, c in zip(labs.tolist(), cnt.tolist()): row[f"vol_{l}"] = c * vox        # mm^3 per segmentation label
    return row

t0 = time.time()
with ThreadPoolExecutor(4) as ex: rows = list(ex.map(one, dirs))
df = pd.DataFrame(rows); df["cdr_binary"] = np.where(df.cdr.isna(), np.nan, (df.cdr > 0).astype(float))
print(f"done in {time.time()-t0:.0f}s | with segmentation volumes: {df['vol_1'].notna().sum() if 'vol_1' in df else 0}")
print("labeled (CDR):", int(df.cdr.notna().sum()), "| unlabeled age range:", df[df.cdr.isna()].age.min(), "-", df[df.cdr.isna()].age.max())
vc = [c for c in df.columns if c.startswith("vol_")]; print("segmentation labels found:", vc)
print("SANITY (mean mm^3 per label, expect CSF smallest-ish, GM & WM ~ 400-700k each):"); print(df[vc].mean().round(0).to_string())
print(df[["age", "cdr", "mmse", "etiv", "asf", "nwbv"]].describe().round(2).T.to_string())
df.to_csv(f"{OUT}/features.csv", index=False); up(f"{OUT}/features.csv", "brainage")
