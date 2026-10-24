# ===== CELL 23 — Manuscript pack: Table 1, software versions, key numbers, and publication figures (architecture, brain output, results) =====
# needs, in the same session: Cell 1 (OUT, up, RCLONE vars), Cell 3 (lab), Cell 12 (D, y, N), Cell 15 (rid, names, load_seg, roi_row, sub_dirs, SIDS, ROOT),
# Cell 19 (FR, rad_row, load_t88_full, ST6, fit_pred3). Result CSVs from earlier cells are downloaded from Drive automatically. No GPU needed.
import traceback, zipfile, platform, importlib
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Patch, Rectangle
from matplotlib.lines import Line2D
from scipy import ndimage
PK = f"{OUT}/paper_pack"; IN = f"{OUT}/paper_inputs"; os.makedirs(PK, exist_ok=True); os.makedirs(IN, exist_ok=True)
C = dict(blue="#0072B2", orange="#E69F00", green="#009E73", verm="#D55E00", sky="#56B4E9", yellow="#F0E442", pink="#CC79A7", grey="#6B6B6B", ink="#222222")
FAM = {"Whole-brain baseline": C["grey"], "Tissue-block features": C["sky"], "3D CNN": C["blue"], "Regional features": C["green"], "Ensemble": C["pink"]}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": .8, "axes.edgecolor": "#444444", "xtick.color": "#444444", "ytick.color": "#444444",
                     "axes.labelcolor": C["ink"], "text.color": C["ink"], "savefig.dpi": 300, "pdf.fonttype": 42, "font.family": "DejaVu Sans"})
def save(fig, name):
    fig.savefig(f"{PK}/{name}.png", bbox_inches="tight", dpi=300); fig.savefig(f"{PK}/{name}.pdf", bbox_inches="tight"); plt.close(fig); print("saved", name)
def down(sub, local, extra=()):
    os.makedirs(local, exist_ok=True)
    r = subprocess.run(["rclone", "copy", f"{RCLONE_REMOTE}:{DRIVE_SUBDIR}/{sub}", local, f"--drive-root-folder-id={DRIVE_FOLDER_ID}", "--transfers=4", "-q", *extra], capture_output=True, text=True)
    print(("downloaded " if r.returncode == 0 else "!! FAILED ") + f"{sub}", r.stderr[:200] if r.returncode else "")
def safe(name, fn):
    try: fn()
    except Exception: traceback.print_exc(); print("!! FAILED:", name)
def inc(*names): return [a for n in names for a in ("--include", n)]
down("paper", IN, inc("table1_main.csv", "table2_paired_deltas.csv"))
down("roi", IN, inc("roi_oof_scores.csv", "sparse_identical_folds.csv", "rad_identical_folds.csv", "roi_ablation.csv", "robust_subsets.csv", "robust_fresh_seeds.csv", "roi_reliability.csv", "sparse_selection_frequency.csv"))
def rd_csv(n): p = f"{IN}/{n}"; return pd.read_csv(p) if os.path.exists(p) else None

# ---------------- 1. Table 1 (LaTeX, auto-included by main.tex) ----------------
def ms(x, d=1):
    x = pd.Series(x).dropna(); return "--" if len(x) == 0 else (f"{x.mean():.{d}f} ({x.std():.{d}f})" if len(x) > 1 else f"{x.mean():.{d}f}")
GRP = [("CDR 0", lab.cdr == 0), ("CDR 0.5", lab.cdr == 0.5), ("CDR 1", lab.cdr == 1), ("CDR 2", lab.cdr == 2), ("All", lab.cdr.notna())]
ROWS = [("Participants, $n$", lambda m: str(int(m.sum()))), ("Age, years", lambda m: ms(lab.age[m])),
        ("Female, $n$ (\\%)", lambda m: ("--" if m.sum()==0 else f"{int(lab.sex_f[m].sum())} ({100*lab.sex_f[m].mean():.0f})")),
        ("Education category (1--5)", lambda m: ms(lab.educ[m])), ("Socioeconomic status (1--5)", lambda m: ms(lab.ses[m])),
        ("MMSE", lambda m: ms(lab.mmse[m])), ("nWBV", lambda m: ms(lab.nwbv[m], 3))]
tex = "\\begin{tabular}{lccccc}\n\\toprule\n & " + " & ".join(g for g, _ in GRP) + "\\\\\n\\midrule\n" + "\n".join(f"{n} & " + " & ".join(f(m) for _, m in GRP) + "\\\\" for n, f in ROWS) + "\n\\bottomrule\n\\end{tabular}\n"
open(f"{PK}/table1_cohort.tex", "w").write(tex); print("\n=== Table 1 (saved as table1_cohort.tex; main.tex inputs it automatically) ===\n" + tex)

# ---------------- 2. Software versions ----------------
vers = {"Python": platform.python_version()}
for m in ["numpy", "pandas", "scipy", "sklearn", "torch", "nibabel", "nilearn", "matplotlib"]:
    try: vers[m] = importlib.import_module(m).__version__
    except Exception: pass
sent = "Software versions: " + ", ".join(f"{k} {v}" for k, v in vers.items()) + "."; open(f"{PK}/software_versions.txt", "w").write(sent + "\n"); print("\n=== " + sent)

# ---------------- 3. Key numbers for the text ----------------
FSTAR = "Hippocampus__r_gm_csf_contact"; fs = np.zeros(N)
for f in range(5): te = (D.fold == f).values; fs[te] = fit_pred3([FSTAR], ~te, te)
from sklearn.metrics import roc_auc_score, roc_curve
sub = {"all (235)": np.arange(N), "CDR0.5 vs 0": np.where(D.cdr.isin([0, 0.5]).values)[0], "CDR>=1 vs 0": np.where(D.cdr.isin([0, 1, 2]).values)[0]}
rng = np.random.default_rng(23); rows = []
for sn, idx in sub.items():
    ys = y[idx]; bs = [b for b in (rng.integers(0, len(idx), len(idx)) for _ in range(1000)) if len(np.unique(ys[b])) == 2]
    for k, s in {"nWBV only": -D.nwbv.values, "F* (hippocampal GM-CSF contact), shared folds": fs}.items():
        v = [roc_auc_score(ys[b], s[idx][b]) for b in bs]; rows.append(dict(subset=sn, n=len(idx), model=k, auroc=roc_auc_score(ys, s[idx]), ci_lo=np.percentile(v, 2.5), ci_hi=np.percentile(v, 97.5)))
TS = pd.DataFrame(rows); TS.to_csv(f"{PK}/fstar_stage_auroc.csv", index=False); print("\n=== AUROC by stage (shared folds) ==="); print(TS.round(3).to_string(index=False))
Fv = FR[FSTAR].values; grp = np.where(D.cdr == 0, "CDR 0", np.where(D.cdr == 0.5, "CDR 0.5", "CDR >=1"))
G2 = pd.DataFrame(dict(group=grp, F=Fv, nwbv=D.nwbv.values)).groupby("group").agg(n=("F", "size"), F_mean=("F", "mean"), F_sd=("F", "std"), nwbv_mean=("nwbv", "mean"), nwbv_sd=("nwbv", "std"))
print("\n=== F* and nWBV by group ==="); print(G2.round(3).to_string()); G2.to_csv(f"{PK}/fstar_group_stats.csv")

# ---------------- 4. Figures ----------------
def fig_architecture():
    from matplotlib.patches import Polygon, Circle
    fig, ax = plt.subplots(figsize=(16, 9.4)); ax.set_aspect("equal"); ax.axis("off"); INK = C["ink"]; rs = np.random.default_rng(5)
    def shade(h, f): rgb = np.array(matplotlib.colors.to_rgb(h)); return tuple(rgb * f) if f < 1 else tuple(rgb + (1 - rgb) * (f - 1))
    def cuboid(x, y, w, h, d, col, z=2, ec="#333333", lw=.6):
        dx, dy = d * .62, d * .45
        ax.add_patch(Polygon([(x, y + h), (x + dx, y + h + dy), (x + w + dx, y + h + dy), (x + w, y + h)], fc=shade(col, 1.35), ec=ec, lw=lw, zorder=z))
        ax.add_patch(Polygon([(x + w, y), (x + w + dx, y + dy), (x + w + dx, y + h + dy), (x + w, y + h)], fc=shade(col, .72), ec=ec, lw=lw, zorder=z))
        ax.add_patch(Rectangle((x, y), w, h, fc=col, ec=ec, lw=lw, zorder=z + .1)); return dx, dy
    def arr(x1, y1, x2, y2, rad=0.0, col="#444444", ls="-", lw=1.2, z=5): ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=10, lw=lw, color=col, ls=ls, connectionstyle=f"arc3,rad={rad}", zorder=z, shrinkA=0, shrinkB=0))
    def lane(x, y, w, h, title, tint, head, tc="white"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.6", fc=tint, ec=head, lw=1.1, ls=(0, (5, 3)), zorder=1))
        ax.add_patch(FancyBboxPatch((x + 1.2, y + h - 1.2), len(title) * 1.12 + 3, 3.4, boxstyle="round,pad=0.1,rounding_size=0.8", fc=head, ec="none", zorder=6)); ax.text(x + 2.4, y + h + .5, title, ha="left", va="center", fontsize=9, color=tc, weight="bold", zorder=7)
    def lab(x, y, t, fs=7.4, **k): ax.text(x, y, t, ha=k.pop("ha", "center"), va=k.pop("va", "center"), fontsize=fs, color=k.pop("color", INK), zorder=8, **k)
    def card(x, y, w, h, fc="white", ec="#555555", z=3):
        ax.add_patch(FancyBboxPatch((x + .45, y - .45), w, h, boxstyle="round,pad=0,rounding_size=0.9", fc="black", ec="none", alpha=.16, zorder=z - .2))
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.9", fc=fc, ec=ec, lw=.8, zorder=z))
    def pill(x, y, w, h, text, fc, fs=6.2):
        ax.add_patch(FancyBboxPatch((x + .35, y - .35), w, h, boxstyle="round,pad=0,rounding_size=0.8", fc="black", ec="none", alpha=.18, zorder=3.8))
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.8", fc=fc, ec=shade(fc, .6), lw=.7, zorder=4))
        ax.text(x + w / 2, y + h / 2, text, rotation=90, ha="center", va="center", fontsize=fs, color="white", weight="bold", zorder=5)
    def fmap(x, y, sz, n, g, col, off=.6):
        from matplotlib.colors import LinearSegmentedColormap
        cm = LinearSegmentedColormap.from_list("c", ["#FFFFFF", col, shade(col, .55)])
        for k in range(n - 1, -1, -1):
            ox, oy = x + k * off, y + k * off * .8; a = np.clip(rs.uniform(.25, 1, (g, g)), 0, 1)
            ax.imshow(a, extent=(ox, ox + sz, oy, oy + sz), cmap=cm, vmin=0, vmax=1, interpolation="nearest", zorder=4 - k * .02, aspect="auto")
            ax.add_patch(Rectangle((ox, oy), sz, sz, fc="none", ec="#333333", lw=.6, zorder=4.01 - k * .02))
            for t in np.linspace(0, sz, g + 1)[1:-1]: ax.plot([ox + t] * 2, [oy, oy + sz], color="white", lw=.35, zorder=4.02 - k * .02); ax.plot([ox, ox + sz], [oy + t] * 2, color="white", lw=.35, zorder=4.02 - k * .02)
    def ball(x, y, r, col):
        ax.add_patch(Circle((x + .12, y - .12), r, fc="black", alpha=.2, ec="none", zorder=4.8)); ax.add_patch(Circle((x, y), r, fc=col, ec=shade(col, .55), lw=.7, zorder=5)); ax.add_patch(Circle((x - r * .3, y + r * .3), r * .35, fc="white", alpha=.55, ec="none", zorder=5.1))
    def vec(x, y, n, cell, cols, vertical=False):
        for i in range(n): ax.add_patch(Rectangle((x, y + i * cell) if vertical else (x + i * cell, y), cell * .9, cell * .9, fc=cols[i % len(cols)] if not callable(cols) else cols(i), ec="#555555", lw=.35, zorder=4))
    def sig(x, y, w, h):
        card(x, y, w, h); t = np.linspace(-6, 6, 50); ax.plot(x + (t + 6) / 12 * w, y + h * (.12 + .76 / (1 + np.exp(-t))), color=C["verm"], lw=1.5, zorder=4)
        ax.plot([x, x + w], [y + h / 2] * 2, color="#CCCCCC", lw=.6, ls=":", zorder=3.5)
    def scat(x, y, w, h, fit=None):
        card(x, y, w, h); px = rs.uniform(.12, .88, 22); py = np.clip(px * .8 + .1 + rs.normal(0, .08, 22), .08, .92); ax.scatter(x + px * w, y + py * h, s=5, color=C["blue"], zorder=4)
        if fit == "pca": ax.add_patch(FancyArrowPatch((x + .12 * w, y + .18 * h), (x + .88 * w, y + .86 * h), arrowstyle="-|>", mutation_scale=8, color=C["verm"], lw=1.4, zorder=5))
        if fit == "line": ax.plot([x + .1 * w, x + .9 * w], [y + .18 * h, y + .88 * h], color=C["verm"], lw=1.4, zorder=5)
    def roc(x, y, w, h):
        card(x, y, w, h); t = np.linspace(0, 1, 30)
        for a_, c_ in [(.5, C["grey"]), (.25, C["green"])]: ax.plot(x + t * w, y + t ** a_ * h, color=c_, lw=1.4, zorder=4)
        ax.plot([x, x + w], [y, y + h], color="#BBBBBB", lw=.6, ls=":", zorder=3.5)
    def funnel(x, y, w, h):
        ax.add_patch(Polygon([(x, y + h), (x + w, y + h), (x + w * .58, y + h * .45), (x + w * .58, y), (x + w * .42, y), (x + w * .42, y + h * .45)], fc="#F4F4F4", ec="#555555", lw=.8, zorder=3))
    # ---------- real thumbnails ----------
    T1 = SEGRGB = ROI = ZOOM = None
    try:
        hid = names.index("Hippocampus") + 1; aid = names.index("Amygdala") + 1; vid = names.index("Lateral_Ventricle") + 1
        Hm, Am, Vm = rid == hid, rid == aid, rid == vid; ch, cv = np.argwhere(Hm).mean(0), np.argwhere(Vm).mean(0); y0 = int(round(ch[1])); flip = cv[2] < ch[2]
        g = D[D.cdr == 0]; sid = g.iloc[(g.age - g.age.median()).abs().argsort().iloc[0]].subject_id; im = load_t88_full(sid); sg = load_seg(sid)
        sl = lambda v: (v[:, y0, :].T[::-1] if flip else v[:, y0, :].T)
        lo_, hi_ = np.percentile(im[im > 0], [1, 99.7]); T1 = np.clip((sl(im).astype(float) - lo_) / (hi_ - lo_), 0, 1)
        S = sl(sg); SEGRGB = np.ones(S.shape + (3,)) * .08
        for k, c in {1: C["sky"], 2: C["orange"], 3: "#F2F2F2"}.items(): SEGRGB[S == k] = matplotlib.colors.to_rgb(c)
        ROI = (sl(Hm).astype(float), sl(Am).astype(float)); Cc = sl(((sg == 2) & Hm & ndimage.binary_dilation(sg == 1, ST6)).astype(float))
        pts = np.argwhere(ROI[0] > 0); mid = T1.shape[1] / 2; right = (pts[:, 1] >= mid).sum() >= (pts[:, 1] < mid).sum(); pts = pts[(pts[:, 1] >= mid) == right]
        r0, c0 = pts.min(0) - 5; r1, c1 = pts.max(0) + 5; hw = int(max(r1 - r0, c1 - c0) / 2); cy, cx = int((r0 + r1) / 2), int((c0 + c1) / 2)
        r_a, c_a = max(cy - hw, 0), max(cx - hw, 0); crop = lambda a: a[r_a:cy + hw, c_a:cx + hw]
        ZOOM = (crop(T1), crop(ROI[0]), crop(Cc))
    except Exception: traceback.print_exc()
    tex = lambda: np.clip(rs.normal(.5, .18, (40, 40)), 0, 1)
    def put(img, x, y, w, h, rgb=False, z=2.5):
        if img is None: img, rgb = tex(), False
        ax.imshow(img, extent=(x, x + w, y, y + h), origin="lower", cmap=None if rgb else "gray", vmin=0, vmax=1, interpolation="bilinear", zorder=z, aspect="auto")
    # ---------- input ----------
    cuboid(4, 62, 15, 15, 5, "#777777"); put(T1, 4, 62, 15, 15); lab(11.5, 58.4, "T1 (atlas, 1 mm)", 8)
    cuboid(4, 33, 15, 15, 5, C["orange"]); put(SEGRGB, 4, 33, 15, 15, rgb=True); lab(11.5, 29.4, "FSL FAST", 8)
    for i, (c_, t_) in enumerate([(C["sky"], "CSF"), (C["orange"], "GM"), ("#F2F2F2", "WM")]):
        ax.add_patch(Rectangle((3 + i * 7, 24), 2, 2, fc=c_, ec="#555555", lw=.5, zorder=4)); lab(5.4 + i * 7, 25, t_, 7, ha="left")
    for yc, y1 in ((83, 70), (64.5, 70), (38, 40), (10, 40)): arr(24, y1, 29.3, yc, col="#999999", lw=1)
    # ---------- A ----------
    lane(30, 76, 98, 15, "A   Whole-brain baseline", "#F0F0F0", C["grey"])
    cw = 22; ax.add_patch(Rectangle((34, 79), cw * .22, 5, fc=C["sky"], ec="#555", lw=.6, zorder=4)); ax.add_patch(Rectangle((34 + cw * .22, 79), cw * .42, 5, fc=C["orange"], ec="#555", lw=.6, zorder=4)); ax.add_patch(Rectangle((34 + cw * .64, 79), cw * .36, 5, fc="#F2F2F2", ec="#555", lw=.6, zorder=4))
    ax.plot([34 + cw * .22, 34 + cw], [86.4, 86.4], color=INK, lw=1, zorder=5); ax.plot([34 + cw * .22] * 2 + [34 + cw] * 0, [85.3, 86.4], color=INK, lw=1, zorder=5); ax.plot([34 + cw] * 2, [85.3, 86.4], color=INK, lw=1, zorder=5)
    lab(34 + cw * .61, 88.2, "(GM+WM) / total", 7.2); lab(45, 76.9 + 0.6, "nWBV", 8.4, weight="bold")
    arr(58, 81.5, 66, 81.5)
    for i, t_ in enumerate(["nWBV", "age", "sex", "eTIV"]):
        ax.add_patch(Circle((70 + i * 6.4, 81.5), 2.3, fc="#FFFFFF" if i else "#E6E6E6", ec="#555555", lw=.8, zorder=4)); lab(70 + i * 6.4, 81.5, t_, 5.8)
    arr(92, 81.5, 99, 81.5); sig(100, 78.5, 9, 6.6); lab(104.5, 76.8, "LR", 7.6, weight="bold")
    # ---------- B ----------
    lane(30, 54, 98, 19, "B   Tissue-block features", "#E6F3FB", C["sky"], tc=INK)
    put(T1, 34, 57.5, 11, 11); [ax.plot([34 + k * 11 / 6] * 2, [57.5, 68.5], color="#FFFFFF", lw=.5, alpha=.85, zorder=3) for k in range(1, 6)]; [ax.plot([34, 45], [57.5 + k * 11 / 6] * 2, color="#FFFFFF", lw=.5, alpha=.85, zorder=3) for k in range(1, 6)]
    lab(39.5, 55.6, "8 mm blocks", 7.2); arr(47, 63, 52, 63)
    vec(53, 62, 17, 1.2, lambda i: (C["sky"], C["orange"], "#E8E8E8")[i % 3]); lab(63.5, 59.8, "10,758 features", 7.2)
    arr(74.5, 63, 78, 66.5); arr(74.5, 63, 78, 59.5)
    scat(79, 64, 8, 6.4, "pca"); lab(83, 62.6, "PCA 40", 7); arr(88, 67.2, 92, 67.2); sig(93, 64, 8, 6.4); lab(97, 62.6, "LR", 7.4, weight="bold")
    scat(79, 56, 8, 5.2, "line"); lab(83, 54.7, "Ridge", 7); arr(88, 58.6, 92, 58.6); lab(97.6, 58.6, "age gap", 7.4, weight="bold")
    # ---------- C ----------
    lane(30, 21, 98, 30, "C   3D CNN", "#DCE8F3", C["blue"])
    cuboid(34, 31, 7.5, 7.5, 6, "#777777"); put(T1, 34, 31, 7.5, 7.5); cuboid(35.6, 32.4, 2.0, 2.0, 2.0, C["verm"], z=6, lw=.5); lab(38.7, 27.4, "88×104×88", 6.6)
    arr(45.6, 35, 47.4, 35, lw=1); lab(79, 47.2, "5 blocks:  Conv 3×3×3 ×2 + GroupNorm + ReLU  →  MaxPool;  channels 16 → 192", 7.4, color="#1B5E8C", weight="bold")
    chs = [16, 32, 64, 128, 192]; dims = ["44×52×44", "22×26×22", "11×13×11", "5×6×5", "2×3×2"]; sz = [7.5, 6.3, 5.2, 4.2, 3.4]; gg = [6, 5, 4, 3, 3]; nn = [2, 3, 3, 4, 4]
    cols = ["#7FB4DD", "#4C93CC", "#2A74B5", "#175A98", "#0B4170"]; x = 47.6; TEAL = "#1B9E8F"
    for i in range(5):
        pill(x, 28.8, 1.8, 12.4, "Conv3³×2 GN ReLU" if i == 0 else "Conv ×2", TEAL, fs=5.4); x += 2.5
        w_ = sz[i] + (nn[i] - 1) * .6; fmap(x, 35 - sz[i] / 2, sz[i], nn[i], gg[i], cols[i]); lab(x + w_ / 2, 26.6, str(chs[i]), 7.4, weight="bold"); lab(x + w_ / 2, 24.9, dims[i], 5.8, color="#444444")
        x += w_ + .6
        if i < 4: pill(x, 31, 1.5, 8, "MaxPool", C["orange"], fs=5.2); x += 2.4
    arr(x + .2, 35, x + 2.4, 35, lw=1); x += 2.8
    for k in range(10): ax.add_patch(Rectangle((x, 29.5 + k * 1.1), 1.0, 1.0, fc="#FFE9A8", ec="#777", lw=.4, zorder=4))
    lab(x + .5, 26.6, "GAP", 6.8); lab(x + .5, 24.9, "dropout", 5.8, color="#444444")
    nx0 = x + 5.0; ys = np.linspace(30, 41, 5); o1, o2 = (nx0 + 7.5, 39.6), (nx0 + 7.5, 31.4)
    for yj in ys:
        for (ox, oy) in (o1, o2): ax.plot([nx0, ox], [yj, oy], color="#AAAAAA", lw=.4, zorder=4)
    for yj in ys: ball(nx0, yj, .95, "#FFD66B")
    ball(o1[0], o1[1], 1.6, "#F4A6CB"); ball(o2[0], o2[1], 1.6, "#7DB8E8"); lab(o1[0] + 2.0, o1[1], "CDR>0", 6.2, ha="left", weight="bold"); lab(o2[0] + 2.0, o2[1], "age", 6.2, ha="left", weight="bold")
    ax.plot([o2[0], o2[0], 33.2], [o2[1] - 1.9, 21.9, 21.9], color=C["verm"], ls=(0, (4, 3)), lw=1.2, zorder=6); arr(33.2, 21.9, 33.2, 30.6, col=C["verm"], ls=(0, (4, 3)), lw=1.2, z=6)
    lab(76, 23.3, "stage 1: age (normals)  →  weights initialise stage 2: CDR classifier", 7, color=C["verm"])
    # ---------- D ----------
    lane(30, 1, 98, 17, "D   Regional features", "#DDF1EA", C["green"])
    put(T1, 34, 3.5, 11, 11)
    if ROI is not None:
        n0, n1 = ROI[0].shape; xs_ = 34 + (np.arange(n1) + .5) / n1 * 11; ys_ = 3.5 + (np.arange(n0) + .5) / n0 * 11
        if ROI[0].any(): ax.contour(xs_, ys_, ROI[0], levels=[.5], colors=C["yellow"], linewidths=1.0, zorder=4)
        if ROI[1].any(): ax.contour(xs_, ys_, ROI[1], levels=[.5], colors=C["sky"], linewidths=1.0, zorder=4)
    lab(39.5, 2.2, "6 ROIs", 7.2); arr(47, 9, 51, 9)
    hm = np.clip(rs.uniform(.15, 1, (6, 16)), 0, 1); ax.imshow(hm, extent=(52, 76, 4.2, 13.8), cmap="Blues", vmin=0, vmax=1.2, interpolation="nearest", zorder=3, aspect="auto"); ax.add_patch(Rectangle((52, 4.2), 24, 9.6, fc="none", ec="#333", lw=.7, zorder=4)); ax.add_patch(Rectangle((52, 4.2 + 9.6 * .75), 24 * .25, 9.6 * .25, fc="none", ec=C["orange"], lw=1.3, zorder=4.1)); lab(64, 2.6, "96 features (24 volume + 72 radiomic)", 6.8)
    arr(80, 9, 83.5, 9); funnel(84, 4, 8, 10); lab(88, 2.6, "top-k (in-fold)", 6.8); arr(93, 9, 96.5, 9); sig(97, 5.8, 8, 6.4); lab(101, 4.2, "LR", 7.4, weight="bold")
    arr(106, 9, 109.5, 9)
    if ZOOM is not None:
        put(ZOOM[0], 110, 3, 14, 14, z=2.5); ov = np.zeros(ZOOM[2].shape + (4,)); ov[ZOOM[2] > 0] = [*matplotlib.colors.to_rgb(C["verm"]), .95]; ax.imshow(ov, extent=(110, 124, 3, 17), origin="lower", interpolation="nearest", zorder=3, aspect="auto")
        zy = 3 + (np.arange(ZOOM[1].shape[0]) + .5) / ZOOM[1].shape[0] * 14; zx = 110 + (np.arange(ZOOM[1].shape[1]) + .5) / ZOOM[1].shape[1] * 14
        if ZOOM[1].any(): ax.contour(zx, zy, ZOOM[1], levels=[.5], colors=C["yellow"], linewidths=1.0, zorder=4)
    else: put(None, 110, 3, 14, 14)
    ax.add_patch(Rectangle((110, 3), 14, 14, fc="none", ec=C["green"], lw=1.4, zorder=6)); lab(117, 1.7, "F*: GM–CSF contact", 7.2, color=C["green"], weight="bold")
    # ---------- evaluation ----------
    lane(134, 30, 25, 61, "Evaluation", "#FFFFFF", "#444444")
    for i in range(5): ax.add_patch(Rectangle((137.5 + i * 3.5, 83), 3.1, 3.1, fc=C["verm"] if i == 2 else "#DDDDDD", ec="#555", lw=.6, zorder=3))
    lab(146.5, 80.6, "subject-level CV", 7.2)
    hx = np.linspace(138, 155, 9); hv = [1, 2.3, 4, 5.6, 4.6, 3.2, 1.9, 1.0, .4]
    for a_, b_ in zip(hx, hv): ax.add_patch(Rectangle((a_ - .8, 69), 1.6, b_ * .9, fc=C["sky"], ec="#555", lw=.4, zorder=3))
    lab(146.5, 66.4, "bootstrap, 2000", 7.2)
    roc(140, 49.5, 13, 13); lab(146.5, 47.4, "AUROC vs nWBV", 7.2)
    scat(140, 33.5, 13, 9.6, "line"); lab(146.5, 31.4, "ICC, 20 repeats", 7.2)
    for yc in (83, 64.5, 38, 10): arr(128.6, yc, 133.4, min(max(yc, 34), 86), col="#999999", lw=1)
    ax.set_xlim(0, 160); ax.set_ylim(0, 94); save(fig, "Fig1_architecture")

def fig_forest():
    a, s1, s2 = rd_csv("table1_main.csv"), rd_csv("sparse_identical_folds.csv"), rd_csv("rad_identical_folds.csv")
    T = pd.concat([t for t in (a, s1, s2) if t is not None]).drop_duplicates("model").set_index("model")
    SPEC = [("nWBV only", "nWBV alone", "Whole-brain baseline"), ("LR age+sex+eTIV+nWBV", "Age + sex + eTIV + nWBV", "Whole-brain baseline"),
            ("Blocks MRI-only (PCA+LR)", "Block features (PCA + LR)", "Tissue-block features"), ("Block-ridge BAG (raw)", "Block-ridge brain-age gap", "Tissue-block features"),
            ("CNN scratch", "CNN from scratch", "3D CNN"), ("CNN age-pretrained", "CNN age-pretrained", "3D CNN"), ("CNN BAG (raw)", "CNN brain-age gap", "3D CNN"),
            ("ROI MTL+vent (MRI-only)", "Regional volumes, six regions", "Regional features"), ("VOL hippocampus+amygdala (post-hoc, Cell 17)", "Hippocampus + amygdala volumes (post hoc)", "Regional features"),
            ("VOL+RAD all 6 ROIs (PRIMARY)", "Volume + radiomic-style, six regions", "Regional features"), ("Top-3 of 96 (in-fold selection) (PRIMARY)", "Sparse top-3 (in-fold selection)", "Regional features"),
            ("Top-1 of 96 (in-fold selection)", "Sparse top-1 = F*  (hippocampal GM\u2013CSF contact)", "Regional features"),
            ("Ensemble (rank-mean of ROI MTL+vent, CNN age-pretrained, Blocks)", "Ensemble (regional + CNN + block)", "Ensemble")]
    SPEC = [s for s in SPEC if s[0] in T.index]; fig, ax = plt.subplots(figsize=(9.2, 5.6)); yy = np.arange(len(SPEC))[::-1]; ref = T.loc["nWBV only", "auroc"]
    for yk, (k, lab_, fam) in zip(yy, SPEC):
        r = T.loc[k]; ax.plot([r.ci_lo, r.ci_hi], [yk, yk], color=FAM[fam], lw=1.8, solid_capstyle="round"); ax.plot(r.auroc, yk, "o", ms=6.5, color=FAM[fam], mec="white", mew=1.2)
        ax.text(0.935, yk, f"{r.auroc:.3f} ({r.ci_lo:.3f}\u2013{r.ci_hi:.3f})", va="center", fontsize=8, color="#333333")
    ax.axvline(ref, color=C["verm"], lw=1, ls="--"); ax.axvline(0.5, color="#BBBBBB", lw=.8); ax.set_yticks(yy); ax.set_yticklabels([s[1] for s in SPEC], fontsize=8.5); ax.set_xlim(0.5, 1.03)
    ax.set_xlabel("Out-of-fold AUROC, CDR > 0 vs CDR = 0 (95% subject-bootstrap CI)"); ax.grid(axis="x", color="#EEEEEE", lw=.6); ax.set_axisbelow(True)
    ax.legend(handles=[Patch(color=v, label=k) for k, v in FAM.items() if any(s[2] == k for s in SPEC)] + [Line2D([0], [0], color=C["verm"], ls="--", label="nWBV alone")], fontsize=8, loc="lower left", frameon=False)
    save(fig, "Fig2_forest")

def fig_ablation():
    T = rd_csv("roi_ablation.csv"); T = T.sort_values("auroc_mean"); fig, ax = plt.subplots(figsize=(7.4, 4.4)); yy = np.arange(len(T)); ref = float(T.loc[T.model == "nWBV only", "auroc_mean"].iloc[0])
    for yk, (_, r) in zip(yy, T.iterrows()):
        col = C["grey"] if r.model == "nWBV only" else (C["green"] if r.auroc_mean >= ref else C["orange"])
        ax.plot([r.auroc_mean - r.auroc_sd, r.auroc_mean + r.auroc_sd], [yk, yk], color=col, lw=1.8); ax.plot(r.auroc_mean, yk, "o", ms=6.5, color=col, mec="white", mew=1.2)
        ax.text(0.905, yk, f"{r.auroc_mean:.3f}", va="center", fontsize=8)
    ax.axvline(ref, color=C["verm"], lw=1, ls="--"); ax.set_yticks(yy); ax.set_yticklabels([m.replace("MTL+vent full (Cell 16)", "All six regions (24 features)").replace("only ", "").replace("nWBV only", "nWBV alone") + (f"  [{int(n)}]" if m != "nWBV only" else "") for m, n in zip(T.model, T.n_features)], fontsize=8.5)
    ax.set_xlim(0.6, 0.93); ax.set_xlabel("AUROC (repeated 5\u00d75 CV, mean \u00b1 SD over repeats); [n features]"); ax.grid(axis="x", color="#EEEEEE", lw=.6); ax.set_axisbelow(True)
    ax.legend(handles=[Line2D([0], [0], marker="o", color=C["green"], lw=0, label="above nWBV"), Line2D([0], [0], marker="o", color=C["orange"], lw=0, label="below nWBV"), Line2D([0], [0], color=C["verm"], ls="--", label="nWBV alone")], fontsize=8, loc="lower right", frameon=False)
    save(fig, "Fig3_ablation")

def fig_roc_and_age():
    O = rd_csv("roi_oof_scores.csv").set_index("subject_id").loc[D.subject_id]; from scipy.stats import rankdata
    sc = {"nWBV alone": (-D.nwbv.values, C["grey"], "-"), "Age + sex + eTIV + nWBV": (O["LR age+sex+eTIV+nWBV"].values, C["sky"], "-"), "CNN age-pretrained": (O["CNN age-pretrained"].values, C["blue"], "-"),
          "Regional volumes, six regions": (O["ROI MTL+vent (MRI-only)"].values, C["green"], "--"), "F* (hippocampal GM\u2013CSF contact)": (fs, C["green"], "-")}
    ens = np.mean([rankdata(O[k].values) for k in ["ROI MTL+vent (MRI-only)", "CNN age-pretrained", "Blocks MRI-only (PCA+LR)"]], 0); sc["Ensemble"] = (ens, C["pink"], "-")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.8), gridspec_kw=dict(width_ratios=[1, 1.15]))
    for k, (s, col, ls) in sc.items(): fpr, tpr, _ = roc_curve(y, s); a1.plot(fpr, tpr, color=col, ls=ls, lw=1.7, label=f"{k} ({roc_auc_score(y, s):.3f})")
    a1.plot([0, 1], [0, 1], color="#BBBBBB", lw=.8, ls=":"); a1.set_xlabel("1 \u2013 specificity"); a1.set_ylabel("Sensitivity"); a1.legend(fontsize=7.6, loc="lower right", frameon=False); a1.set_aspect("equal"); a1.set_title("a   ROC curves, shared folds", loc="left", fontsize=10, weight="bold")
    S = rd_csv("robust_subsets.csv"); MODS = [("nWBV only", "nWBV alone", C["grey"]), ("age+sex+eTIV+nWBV (clinical covariates)", "Age + sex + eTIV + nWBV", C["sky"]), ("F* alone (fixed feature)", "F* alone", C["green"]), ("age+sex+eTIV+nWBV + F*", "Covariates + F*", C["verm"])]
    SUBS = [("all", "All"), ("CDR0.5_vs_0", "CDR 0.5 vs 0"), ("age-matched(2y)", "Age-matched \u00b12 y")]; off = np.linspace(-.27, .27, len(MODS))
    for j, (mk, ml, col) in enumerate(MODS):
        xs, ys, lo, hi = [], [], [], []
        for i, (sk, _) in enumerate(SUBS):
            r = S[(S["subset"] == sk) & (S.model == mk)].iloc[0]; xs.append(i + off[j]); ys.append(r.auroc); lo.append(r.auroc - r.ci_lo); hi.append(r.ci_hi - r.auroc)
        a2.errorbar(xs, ys, yerr=[lo, hi], fmt="o", ms=6, color=col, mec="white", mew=1.1, capsize=0, lw=1.6, label=ml)
    ns = {sk: int(S[S["subset"] == sk].n.iloc[0]) for sk, _ in SUBS}; a2.set_xticks(range(len(SUBS))); a2.set_xticklabels([f"{l}\n(n = {ns[k]})" for k, l in SUBS]); a2.set_ylabel("AUROC (95% CI)"); a2.set_ylim(0.55, 0.95)
    a2.axhline(0.5, color="#BBBBBB", lw=.8); a2.grid(axis="y", color="#EEEEEE", lw=.6); a2.set_axisbelow(True); a2.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2); a2.set_title("b   Effect of stage and age matching", loc="left", fontsize=10, weight="bold")
    save(fig, "Fig4_roc_and_age_matching")

def fig_retest():
    mr2 = sorted(glob.glob(f"{ROOT}/OAS1_*_MR2"))
    for d in mr2: sub_dirs[os.path.basename(d)] = d
    pairs = [(os.path.basename(d)[:-1] + "1", os.path.basename(d)) for d in mr2 if os.path.basename(d)[:-1] + "1" in set(SIDS)]
    def nw(sid): s_ = load_seg(sid); c_, g_, w_ = [(s_ == k).sum() for k in (1, 2, 3)]; return (g_ + w_) / (c_ + g_ + w_)
    def feats(sid): r = roi_row(sid); r["nWBV (from segmentation)"] = nw(sid); r.update({k: v for k, v in rad_row(sid).items() if k.endswith("gm_csf_contact")}); return r
    F1 = pd.DataFrame([feats(a) for a, b in pairs]); F2 = pd.DataFrame([feats(b) for a, b in pairs]); n = len(pairs)
    def icc21(a, b):
        X = np.column_stack([a, b]); k = 2; gm = X.mean(); msr = k * ((X.mean(1) - gm) ** 2).sum() / (n - 1); msc = n * ((X.mean(0) - gm) ** 2).sum() / (k - 1)
        mse = ((X - X.mean(1, keepdims=True) - X.mean(0, keepdims=True) + gm) ** 2).sum() / ((n - 1) * (k - 1)); return (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)
    P = [("Hippocampus__r_gm_csf_contact", "F*: hippocampal GM\u2013CSF contact"), ("Hippocampus__csf_frac", "Hippocampal CSF fraction"), ("Amygdala__gm_frac", "Amygdalar GM fraction"), ("nWBV (from segmentation)", "nWBV")]
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.6))
    for ax, (k, t) in zip(axs, P):
        a, b = F1[k].values.astype(float), F2[k].values.astype(float); lo, hi = min(a.min(), b.min()), max(a.max(), b.max()); pad = .05 * (hi - lo)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#BBBBBB", lw=1); ax.scatter(a, b, s=34, color=C["blue"], edgecolor="white", lw=.8, zorder=3)
        ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(lo - pad, hi + pad); ax.set_aspect("equal"); ax.set_title(t, fontsize=9.5, loc="left"); ax.set_xlabel("Session 1"); ax.text(.04, .94, f"ICC(2,1) = {icc21(a, b):.3f}\nRC = {2.77*np.sqrt(np.mean((b-a)**2)/2):.4f}", transform=ax.transAxes, va="top", fontsize=8.3)
    axs[0].set_ylabel("Session 2"); fig.suptitle(f"Test\u2013retest agreement, {n} repeat pairs (ages 19\u201334 y); grey line = identity", fontsize=9.5, x=0.01, ha="left", y=1.02); plt.tight_layout(); save(fig, "Fig5_test_retest")

def fig_brain():
    hid, aid, vid = names.index("Hippocampus") + 1, names.index("Amygdala") + 1, names.index("Lateral_Ventricle") + 1
    H, A, V = rid == hid, rid == aid, rid == vid; ch, ca, cv = [np.argwhere(m).mean(0) for m in (H, A, V)]
    flip_z = cv[2] < ch[2]; flip_y = ca[1] < ch[1]                                    # orient so that superior is up (ventricle above hippocampus) and anterior is up (amygdala in front of hippocampus)
    y0, z0 = int(round(ch[1])), int(round(ch[2])); Ldf = D.assign(F=Fv)
    g1 = Ldf[Ldf.cdr == 1] if (Ldf.cdr == 1).sum() else Ldf[Ldf.cdr >= 1]; b = g1.iloc[(g1.F - g1.F.median()).abs().argsort().iloc[0]]; picks = {}
    for lab_, c_ in [("CDR 0", Ldf.cdr == 0), ("CDR 0.5", Ldf.cdr == 0.5)]:
        g = Ldf[c_]
        for tol in (4, 6, 10, 99):
            gg = g[(g.age - b.age).abs() <= tol]
            if len(gg) >= 3: break
        picks[lab_] = gg.iloc[(gg.F - gg.F.median()).abs().argsort().iloc[0]]
    picks[f"CDR {b.cdr:g}"] = b; order = [k for k in picks if k == "CDR 0"] + [k for k in picks if k == "CDR 0.5"] + [k for k in picks if k not in ("CDR 0", "CDR 0.5")]
    print("\n=== Brain figure subjects ==="); [print(f"  {k}: {picks[k].subject_id}  age {picks[k].age:.0f}  F* {picks[k].F:.3f}") for k in order]
    def view(img2, flip, T=True): a = img2.T if T else img2; return a[::-1] if flip else a
    fig, axs = plt.subplots(len(order), 3, figsize=(10.2, 3.25 * len(order))); axs = np.atleast_2d(axs)
    for r_, k in enumerate(order):
        s = picks[k]; sid = s.subject_id; im = load_t88_full(sid); sg = load_seg(sid); contact = (sg == 2) & H & ndimage.binary_dilation(sg == 1, ST6); brain = im > 0
        lo_, hi_ = np.percentile(im[brain], [1, 99.7])
        for c_, kind in enumerate(["coronal", "axial", "zoom"]):
            ax = axs[r_, c_]
            sl = (slice(None), y0, slice(None)) if kind != "axial" else (slice(None), slice(None), z0); fl = flip_z if kind != "axial" else flip_y
            I, Hs, As, Cs = [view(m[sl], fl) for m in (im, H, A, contact)]
            ax.imshow(np.clip((I - lo_) / (hi_ - lo_), 0, 1), cmap="gray", origin="lower", interpolation="nearest")
            ov = np.zeros(I.shape + (4,)); ov[Cs] = [*matplotlib.colors.to_rgb(C["verm"]), .95]; ax.imshow(ov, origin="lower", interpolation="nearest")
            if Hs.any(): ax.contour(Hs.astype(float), levels=[.5], colors=C["yellow"], linewidths=.9)
            if As.any(): ax.contour(As.astype(float), levels=[.5], colors=C["sky"], linewidths=.9)
            if kind == "zoom":
                mid = I.shape[1] / 2; cols = np.argwhere(Hs)[:, 1]; right = (cols >= mid).sum() >= (cols < mid).sum()
                pts = np.argwhere(Hs | As); pts = pts[(pts[:, 1] >= mid) == right]
                if len(pts):
                    r0, c0 = pts.min(0) - 8; r1, c1 = pts.max(0) + 8; cy, cx = (r0 + r1) / 2, (c0 + c1) / 2; hw = max(r1 - r0, c1 - c0) / 2
                    ax.set_xlim(cx - hw, cx + hw); ax.set_ylim(cy - hw, cy + hw)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_visible(False)
            if r_ == 0: ax.set_title({"coronal": "Coronal", "axial": "Axial", "zoom": "Coronal, medial temporal zoom"}[kind], fontsize=10)
            if c_ == 0: ax.set_ylabel(f"{k}\n{s.age:.0f} y\nF* = {s.F:.3f}", fontsize=10, rotation=0, ha="right", va="center", labelpad=8)
    fig.legend(handles=[Line2D([0], [0], color=C["yellow"], lw=2, label="Hippocampus (atlas ROI)"), Line2D([0], [0], color=C["sky"], lw=2, label="Amygdala (atlas ROI)"), Patch(color=C["verm"], label="Hippocampal grey matter touching CSF (F* numerator)")],
               loc="lower center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.01)); fig.tight_layout(rect=(0.07, 0.03, 1, 1)); save(fig, "Fig6_brain_output")

def fig_groups():
    cols = {"CDR 0": "#9ECAE1", "CDR 0.5": "#4292C6", "CDR >=1": "#08519C"}; fig, axs = plt.subplots(1, 2, figsize=(8, 3.8)); rng = np.random.default_rng(1)
    for ax, (v, t) in zip(axs, [(Fv, "F*: hippocampal GM\u2013CSF contact"), (D.nwbv.values, "nWBV")]):
        for i, g in enumerate(cols):
            x = v[grp == g]; ax.scatter(i + rng.uniform(-.17, .17, len(x)), x, s=14, color=cols[g], alpha=.75, edgecolor="none", zorder=2)
            ax.plot([i - .28, i + .28], [np.median(x)] * 2, color=C["ink"], lw=2, zorder=3)
        ax.set_xticks(range(3)); ax.set_xticklabels([f"{g}\n(n = {int((grp == g).sum())})" for g in cols]); ax.set_title(t, fontsize=10, loc="left"); ax.grid(axis="y", color="#EEEEEE", lw=.6); ax.set_axisbelow(True)
    save(fig, "Fig7_group_distributions")

for nm, fn in [("Fig1 architecture", fig_architecture), ("Fig2 forest", fig_forest), ("Fig3 ablation", fig_ablation), ("Fig4 ROC + age matching", fig_roc_and_age),
               ("Fig5 test-retest", fig_retest), ("Fig6 brain output", fig_brain), ("Fig7 group distributions", fig_groups)]: safe(nm, fn)

# ---------------- 5. zip and upload ----------------
with zipfile.ZipFile(f"{OUT}/paper_pack.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for f in sorted(os.listdir(PK)): z.write(f"{PK}/{f}", f)
print("\nfiles:", sorted(os.listdir(PK))); up(PK, "paper/figures_and_tables"); up(f"{OUT}/paper_pack.zip", "paper")
