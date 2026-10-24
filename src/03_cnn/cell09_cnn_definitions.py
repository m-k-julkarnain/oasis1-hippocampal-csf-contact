# ===== CELL 9 — Small 3D CNN, augmentation, training / prediction functions =====
BATCH = 8
CNN_DIR = f"{OUT}/cnn"; os.makedirs(CNN_DIR, exist_ok=True)

class Net3D(nn.Module):
    def __init__(s, ch=(16, 32, 64, 128, 192)):
        super().__init__(); L, c_in = [], 1
        for c in ch:
            L += [nn.Conv3d(c_in, c, 3, padding=1, bias=False), nn.GroupNorm(8, c), nn.ReLU(inplace=True),
                  nn.Conv3d(c, c, 3, padding=1, bias=False), nn.GroupNorm(8, c), nn.ReLU(inplace=True), nn.MaxPool3d(2)]
            c_in = c
        s.features = nn.Sequential(*L); s.drop = nn.Dropout(0.3); s.head = nn.Linear(ch[-1], 1)
    def forward(s, x):
        f = F.adaptive_avg_pool3d(s.features(x), 1).flatten(1); return s.head(s.drop(f)).squeeze(1), f

def rot(a):
    B = a.shape[0]; c, s = torch.cos(a), torch.sin(a); o, z = torch.ones(B, device=a.device), torch.zeros(B, device=a.device)
    Rx = torch.stack([o, z, z, z, c[:, 0], -s[:, 0], z, s[:, 0], c[:, 0]], 1).view(B, 3, 3)
    Ry = torch.stack([c[:, 1], z, s[:, 1], z, o, z, -s[:, 1], z, c[:, 1]], 1).view(B, 3, 3)
    Rz = torch.stack([c[:, 2], -s[:, 2], z, s[:, 2], c[:, 2], z, z, z, o], 1).view(B, 3, 3)
    return Rz @ Ry @ Rx

def augment(x):   # conservative: rot<=5deg, trans<=3%, scale +-3%, intensity +-10%
    B, d = x.shape[0], x.device
    R = rot((torch.rand(B, 3, device=d) * 2 - 1) * math.radians(5)) * (0.97 + 0.06 * torch.rand(B, 1, 1, device=d))
    th = torch.cat([R, (torch.rand(B, 3, 1, device=d) * 2 - 1) * 0.03], 2)
    x = F.grid_sample(x, F.affine_grid(th, x.shape, align_corners=False), padding_mode="zeros", align_corners=False)
    return x * (0.9 + 0.2 * torch.rand(B, 1, 1, 1, 1, device=d))

@torch.no_grad()
def predict_raw(model, sids):
    model.eval(); out = []
    for k in range(0, len(sids), 16):
        x = IMG[[SIDX[s] for s in sids[k:k + 16]]].float()
        with torch.autocast(DEVICE, dtype=torch.float16, enabled=DEVICE == "cuda"): o, _ = model(x)
        out.append(o.float().cpu().numpy())
    return np.concatenate(out)

def fit(task, tr_ids, va_ids, target, init=None, epochs=60, lr=5e-4, patience=15, seed=0, tag="m"):
    """task 'age': smooth-L1 on standardised age, select by val MAE. task 'cdr': weighted BCE, select by val loss."""
    seed_all(seed); y_tr = np.array([target[s] for s in tr_ids], dtype=np.float32); y_va = np.array([target[s] for s in va_ids], dtype=np.float32)
    if task == "age": mu, sd = float(y_tr.mean()), float(y_tr.std()) + 1e-6; posw = None
    else: mu, sd = 0.0, 1.0; posw = torch.tensor(float((len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1)), device=DEVICE)
    model = Net3D().to(DEVICE)
    if init is not None: model.features.load_state_dict(init)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2); spe = math.ceil(len(tr_ids) / BATCH)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * spe, pct_start=0.15)
    scaler = torch.cuda.amp.GradScaler(enabled=DEVICE == "cuda"); best, bad, best_state, hist = 1e9, 0, None, []
    for ep in range(epochs):
        model.train(); perm = np.random.permutation(len(tr_ids)); tl = 0.0
        for k in range(0, len(perm), BATCH):
            ii = perm[k:k + BATCH]
            x = augment(IMG[[SIDX[tr_ids[i]] for i in ii]].float()); y = torch.tensor((y_tr[ii] - mu) / sd, device=DEVICE)
            with torch.autocast(DEVICE, dtype=torch.float16, enabled=DEVICE == "cuda"): o, _ = model(x)
            loss = F.smooth_l1_loss(o.float(), y) if task == "age" else F.binary_cross_entropy_with_logits(o.float(), y, pos_weight=posw)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 2.0); scaler.step(opt); scaler.update(); sched.step(); tl += loss.item() * len(ii)
        pv = predict_raw(model, va_ids)
        if task == "age": vm = float(np.abs(pv * sd + mu - y_va).mean()); vauc = np.nan
        else:
            vm = float(F.binary_cross_entropy_with_logits(torch.tensor(pv), torch.tensor(y_va), pos_weight=posw.cpu()).item())
            vauc = roc_auc_score(y_va, pv) if len(np.unique(y_va)) > 1 else np.nan
        hist.append(dict(epoch=ep, train_loss=tl / len(tr_ids), val_metric=vm, val_auroc=vauc))
        if ep % 10 == 0 or ep == epochs - 1: print(f"  {tag} ep{ep:3d} train {tl/len(tr_ids):.3f} val {'MAE' if task=='age' else 'loss'} {vm:.3f}" + ("" if task == "age" else f" auc {vauc:.3f}"))
        if vm < best: best, bad, best_state = vm, 0, copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= patience: print(f"  {tag} early stop at ep{ep}"); break
    model.load_state_dict(best_state); torch.save({"state": best_state, "mu": mu, "sd": sd}, f"{CNN_DIR}/{tag}.pt")
    return model, mu, sd, pd.DataFrame(hist), best
