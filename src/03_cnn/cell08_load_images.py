# ===== CELL 8 — Load T88 masked images (atlas space, skull-stripped), downsample to 2 mm, cache on GPU =====
import torch, torch.nn as nn, torch.nn.functional as F, math, random, copy
from concurrent.futures import ThreadPoolExecutor
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"; print("device:", DEVICE, torch.__version__)
if "ROOT" not in globals():
    ROOT = pd.Series([os.path.dirname(p) for p in glob.glob(f"{DATA_DIR}/**/OAS1_*_MR1", recursive=True) if os.path.isdir(p)]).value_counts().index[0]
sub_dirs = {os.path.basename(d): d for d in sorted(glob.glob(f"{ROOT}/OAS1_*_MR1"))}
def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def load_t88(sid):
    f = glob.glob(f"{sub_dirs[sid]}/PROCESSED/MPRAGE/T88_111/*_t88_masked_gfc.img")[0]
    a = np.asarray(nib.load(f).dataobj).squeeze().astype(np.float32)           # 176 x 208 x 176, 1 mm
    t = F.avg_pool3d(torch.from_numpy(a)[None, None], 2)[0]                     # -> 88 x 104 x 88, 2 mm
    m = t > 0; hi = torch.quantile(t[m][::5], 0.995)
    t = t.clamp(0, hi) / hi; mu, sd = t[m].mean(), t[m].std()
    return torch.where(m, (t - mu) / (sd + 1e-6), torch.zeros_like(t)).half()   # z-score inside brain, 0 outside

SIDS = list(pd.read_csv(f"{OUT}/features.csv").subject_id); SIDX = {s: i for i, s in enumerate(SIDS)}
t0 = time.time()
with ThreadPoolExecutor(4) as ex: vols = list(ex.map(load_t88, SIDS))
IMG = torch.stack(vols).to(DEVICE); del vols
print(f"IMG {tuple(IMG.shape)} fp16 {IMG.numel()*2/1e9:.2f} GB on {DEVICE} | {time.time()-t0:.0f}s")
