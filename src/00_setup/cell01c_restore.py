# ===== CELL 1C — Fast restore of intermediate files from Google Drive (run after Cell 1 instead of Cells 2, 5-11, 13-15) =====
# Needs Cell 1 (OUT, up, RCLONE_REMOTE, DRIVE_SUBDIR, DRIVE_FOLDER_ID). Downloads the intermediate tables written by earlier cells.
def _down(sub, patterns):
    os.makedirs(OUT, exist_ok=True)
    args = ["rclone", "copy", f"{RCLONE_REMOTE}:{DRIVE_SUBDIR}/{sub}", OUT, f"--drive-root-folder-id={DRIVE_FOLDER_ID}", "--transfers=4", "-q"]
    for p in patterns: args += ["--include", p]
    r = subprocess.run(args, capture_output=True, text=True)
    print(("restored " if r.returncode == 0 else "!! FAILED ") + f"{sub}: {patterns}", r.stderr[:300] if r.returncode else "")
_down("brainage", ["features.csv", "blocks.npz"])
_down("roi", ["roi_features.csv", "roi_sets.json"])
_down("cnn", ["cnn_fold*.csv"])
missing = [f for f in ["features.csv", "blocks.npz", "roi_features.csv", "roi_sets.json"] + [f"cnn_fold{i}.csv" for i in range(5)] if not os.path.exists(f"{OUT}/{f}")]
print("all intermediate files present" if not missing else f"!! still missing: {missing}")
