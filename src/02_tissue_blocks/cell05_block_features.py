# ===== CELL 5 — Atlas-space block tissue features (8 mm blocks of CSF/GM/WM fractions) =====
from concurrent.futures import ThreadPoolExecutor
if "ROOT" not in globals():
    ROOT = pd.Series([os.path.dirname(p) for p in glob.glob(f"{DATA_DIR}/**/OAS1_*_MR1", recursive=True) if os.path.isdir(p)]).value_counts().index[0]
BLK = 8
sub_dirs = {os.path.basename(d): d for d in sorted(glob.glob(f"{ROOT}/OAS1_*_MR1"))}
def blocks(sid):
    f = glob.glob(f"{sub_dirs[sid]}/FSL_SEG/*_fseg.img")[0]
    a = np.asarray(nib.load(f).dataobj).squeeze().astype(np.uint8)
    nx, ny, nz = [s // BLK for s in a.shape]
    a = a[:nx * BLK, :ny * BLK, :nz * BLK].reshape(nx, BLK, ny, BLK, nz, BLK)
    return np.stack([(a == k).mean(axis=(1, 3, 5)) for k in (1, 2, 3)], -1).astype(np.float16)   # (nx,ny,nz,3) = CSF,GM,WM
sids = list(pd.read_csv(f"{OUT}/features.csv").subject_id)
t0 = time.time()
with ThreadPoolExecutor(4) as ex: arr = list(ex.map(blocks, sids))
Xb = np.stack(arr).astype(np.float32); keep = Xb.sum(-1).mean(0) > 0.2          # blocks that are inside the brain in most subjects
X = Xb[:, keep, :].reshape(len(sids), -1)
print(f"blocks grid {Xb.shape[1:4]} | kept {int(keep.sum())} blocks -> {X.shape[1]} features | {time.time()-t0:.0f}s")
np.savez_compressed(f"{OUT}/blocks.npz", X=X, sids=np.array(sids), keep=keep); up(f"{OUT}/blocks.npz", "brainage")
