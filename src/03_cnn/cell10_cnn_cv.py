# ===== CELL 10 — 5-fold subject-level CV: (1) normal-only brain-age CNN, (2) CDR classifier: scratch vs age-pretrained =====
# needs Cell 1, Cell 3 (lab, unl, imports), Cell 8, Cell 9 in the same session. Folds are saved/resumable.
from sklearn.model_selection import StratifiedKFold, train_test_split
AGE_EPOCHS, CDR_EPOCHS = 80, 50
AGE = dict(zip(df.subject_id, df.age)); CDRB = dict(zip(lab.subject_id, lab.cdr_binary.astype(float)))
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0); all_rows = []
for fold, (tr, te) in enumerate(skf.split(lab, lab.cdr_binary)):
    fcsv = f"{OUT}/cnn_fold{fold}.csv"
    if os.path.exists(fcsv): all_rows.append(pd.read_csv(fcsv)); print("skip finished fold", fold); continue
    t0 = time.time(); trL, teL = lab.iloc[tr], lab.iloc[te]; te_ids = list(teL.subject_id)
    # ---- stage 1: brain-age model trained ONLY on presumed-normal subjects of the training fold (CDR0 + young unlabeled)
    norm = pd.concat([trL[trL.cdr == 0], unl], ignore_index=True).subject_id.values
    n_tr, n_va = train_test_split(norm, test_size=0.15, random_state=fold)
    am, amu, asd, ah, ab = fit("age", list(n_tr), list(n_va), AGE, epochs=AGE_EPOCHS, lr=1e-3, patience=20, seed=fold, tag=f"age_f{fold}")
    pa_te = predict_raw(am, te_ids) * asd + amu; pa_va = predict_raw(am, list(n_va)) * asd + amu
    a_, b_ = np.polyfit(np.array([AGE[s] for s in n_va]), pa_va, 1)          # bias line from held-out normal validation subjects
    if a_ < 0.2: a_, b_ = 1.0, 0.0
    init = {k: v.clone() for k, v in am.features.state_dict().items()}
    # ---- stage 2: CDR classifier, two arms, identical splits / hyper-parameters
    c_tr, c_va = train_test_split(trL.subject_id.values, test_size=0.2, stratify=trL.cdr_binary.values, random_state=fold); res = {}
    for arm, ini in [("scratch", None), ("agepre", init)]:
        cm, _, _, ch, cb = fit("cdr", list(c_tr), list(c_va), CDRB, init=ini, epochs=CDR_EPOCHS, lr=5e-4, patience=15, seed=fold, tag=f"cdr_{arm}_f{fold}")
        res[arm] = 1 / (1 + np.exp(-predict_raw(cm, te_ids))); del cm
    out = pd.DataFrame(dict(subject_id=te_ids, fold=fold, age_pred=pa_te, p_scratch=res["scratch"], p_agepre=res["agepre"]))
    out["bag_raw"] = pa_te - teL.age.values; out["bag_corr"] = (pa_te - b_) / a_ - teL.age.values
    out.to_csv(fcsv, index=False); all_rows.append(out); del am; torch.cuda.empty_cache()
    n0 = out.merge(lab[["subject_id", "age", "cdr", "cdr_binary"]], on="subject_id"); n0c = n0[n0.cdr == 0]
    print(f"FOLD {fold} done in {(time.time()-t0)/60:.1f} min | val-normal MAE {ab:.2f}y | held-out CDR0 MAE {np.abs(n0c.age_pred-n0c.age).mean():.2f}y | "
          f"AUROC scratch {roc_auc_score(n0.cdr_binary, n0.p_scratch):.3f} agepre {roc_auc_score(n0.cdr_binary, n0.p_agepre):.3f}")
    up(fcsv, "cnn"); up(CNN_DIR, "cnn/checkpoints")
