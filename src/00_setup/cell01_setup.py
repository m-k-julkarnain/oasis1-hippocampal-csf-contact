# ===== CELL 1 — Setup, config, Google Drive (rclone) =====
import os, sys, glob, json, re, time, shutil, subprocess, warnings
warnings.filterwarnings("ignore")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "nibabel"], check=False)
import numpy as np, pandas as pd, nibabel as nib

OUT = "/kaggle/working/oasis_brainage"; os.makedirs(OUT, exist_ok=True)
DATA_DIR        = "/kaggle/input/datasets/mkjulkarnain/oasis1-mri"
RCLONE_REMOTE   = "gdrive"
DRIVE_FOLDER_ID = "1Ue4mDSHzdRnUxR_PKffLt0DM5Fh0-Ngj"
DRIVE_SUBDIR    = "oasis1_brainage"

if not shutil.which("rclone"):      # needs Internet ON in notebook settings
    subprocess.run("cd /tmp && curl -sSL -o rc.zip https://downloads.rclone.org/rclone-current-linux-amd64.zip && "
                   "unzip -oq rc.zip && cp rclone-*-linux-amd64/rclone /usr/local/bin/rclone && chmod +x /usr/local/bin/rclone", shell=True)
try:
    from kaggle_secrets import UserSecretsClient
    sec = UserSecretsClient().get_secret("RCLONE_DRIVE_TOKEN").strip()
    conf = sec if sec.startswith("[") else f"[{RCLONE_REMOTE}]\ntype = drive\nscope = drive\ntoken = {sec}\n"
    os.makedirs(os.path.expanduser("~/.config/rclone"), exist_ok=True)
    open(os.path.expanduser("~/.config/rclone/rclone.conf"), "w").write(conf); print("rclone configured")
except Exception as e:
    print("Secret RCLONE_DRIVE_TOKEN not loaded:", e)

def up(local, sub=""):
    """Copy a file or folder to Drive (never crashes the notebook)."""
    if not shutil.which("rclone"): print("!! rclone not found"); return False
    dest = f"{RCLONE_REMOTE}:{DRIVE_SUBDIR}/{sub}".rstrip("/")
    r = subprocess.run(["rclone", "copy", local, dest, f"--drive-root-folder-id={DRIVE_FOLDER_ID}", "--transfers=4", "-q"],
                       capture_output=True, text=True)
    print(("uploaded " if r.returncode == 0 else "!! UPLOAD FAILED ") + f"{local} -> {dest}", r.stderr[:300] if r.returncode else "")
    return r.returncode == 0

open(f"{OUT}/drive_test.txt", "w").write("ok"); up(f"{OUT}/drive_test.txt")
