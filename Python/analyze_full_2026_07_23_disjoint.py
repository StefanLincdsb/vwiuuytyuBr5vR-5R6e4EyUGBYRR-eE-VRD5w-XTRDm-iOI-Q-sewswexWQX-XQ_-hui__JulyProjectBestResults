from __future__ import annotations

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/pkgs/')
import os, json, math
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # avoid OMP #15 crash (duplicate libiomp5md.dll in this conda env)
import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             recall_score, balanced_accuracy_score,
                             roc_curve, roc_auc_score)

RUNS = "runs_full_disjoint"
ASSETS = "report_assets_disjoint"
SEEDS = [0, 1, 2]
ALPHA = 0.10
os.makedirs(ASSETS, exist_ok=True)


# ----------------------------------------------------------------------------
# small stats helpers
# ----------------------------------------------------------------------------
def softmax_np(z, T=1.0):
    z = z / T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def ece(probs, labels, n_bins=15):
    conf = probs.max(1); pred = probs.argmax(1); correct = (pred == labels).astype(float)
    bins = np.linspace(0, 1, n_bins + 1); n = len(labels); e = 0.0
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.sum() > 0:
            e += (m.sum() / n) * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def fit_temperature(logits, labels, max_iter=200):
    z = torch.tensor(logits, dtype=torch.float32); y = torch.tensor(labels, dtype=torch.long)
    logT = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=max_iter)
    nll = nn.CrossEntropyLoss()
    def closure():
        opt.zero_grad(); loss = nll(z / logT.exp(), y); loss.backward(); return loss
    opt.step(closure)
    return float(logT.exp().item())


def wilson_ci(k, n, z=1.96):
    p = k / n; d = 1 + z**2 / n; c = p + z**2 / (2 * n)
    hw = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return ((c - hw) / d, (c + hw) / d)


def bootstrap_ci(y_true, y_pred, metric, B=2000, seed=0):
    rng = np.random.default_rng(seed); n = len(y_true); vals = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n); vals[b] = metric(y_true[idx], y_pred[idx])
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return metric(y_true, y_pred), float(lo), float(hi)


def mcnemar(y_true, pa, pb):
    a_ok = (pa == y_true); b_ok = (pb == y_true)
    b = int(np.sum(a_ok & ~b_ok)); c = int(np.sum(~a_ok & b_ok)); n = b + c
    if n == 0:
        return dict(b=b, c=c, p_exact=1.0, p_chi2=1.0, chi2=0.0)
    k = min(b, c)
    p_exact = min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n))
    chi2 = (abs(b - c) - 1) ** 2 / n
    p_chi2 = math.erfc(math.sqrt(chi2 / 2.0))
    return dict(b=b, c=c, p_exact=float(p_exact), p_chi2=float(p_chi2), chi2=float(chi2))


def paired_bootstrap_diff(y_true, pa, pb, metric, B=2000, seed=0):
    rng = np.random.default_rng(seed); n = len(y_true); diffs = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        diffs[b] = metric(y_true[idx], pa[idx]) - metric(y_true[idx], pb[idx])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    obs = metric(y_true, pa) - metric(y_true, pb)
    return obs, float(lo), float(hi)


def macrof1(yt, yp): return float(f1_score(yt, yp, average="macro", zero_division=0))
def acc(yt, yp):     return float((yt == yp).mean())


# ----------------------------------------------------------------------------
# load
# ----------------------------------------------------------------------------
def load(seed, kind):
    z = np.load(os.path.join(RUNS, f"seed{seed}_{kind}.npz"), allow_pickle=True)
    return dict(
        test_labels=z["test_labels"], test_pred=z["test_pred"], test_logits=z["test_logits"],
        calib_labels=z["calib_labels"], calib_logits=z["calib_logits"],
        classes=[str(c) for c in z["classes"]],
        history=json.loads(str(z["history_json"])))


def main():
    ft = {s: load(s, "ft") for s in SEEDS}
    lp = {s: load(s, "lp") for s in SEEDS}
    classes = ft[SEEDS[0]]["classes"]; K = len(classes)
    y_true = ft[SEEDS[0]]["test_labels"]; n = len(y_true)

    res = {"classes": classes, "n_test": int(n), "seeds": SEEDS, "per_seed": {}}

    # ---- per-seed metrics -------------------------------------------------
    for s in SEEDS:
        yp = ft[s]["test_pred"]
        T = fit_temperature(ft[s]["calib_logits"], ft[s]["calib_labels"])
        prob_raw = softmax_np(ft[s]["test_logits"])
        prob_cal = softmax_np(ft[s]["test_logits"], T)
        # conformal on calibrated probs
        calib_prob_cal = softmax_np(ft[s]["calib_logits"], T)
        yc = ft[s]["calib_labels"]
        cal_scores = 1.0 - calib_prob_cal[np.arange(len(yc)), yc]
        n_cal = len(cal_scores)
        q_level = min(1.0, math.ceil((n_cal + 1) * (1 - ALPHA)) / n_cal)
        qhat = float(np.quantile(cal_scores, q_level, method="higher"))
        sets = (prob_cal >= (1 - qhat))
        coverage = float(sets[np.arange(n), y_true].mean())
        set_sizes = sets.sum(1)
        # one-vs-rest macro AUC
        auc_macro = float(roc_auc_score(np.eye(K)[y_true], prob_cal, average="macro",
                                        multi_class="ovr"))
        mc = mcnemar(y_true, ft[s]["test_pred"], lp[s]["test_pred"])
        res["per_seed"][s] = dict(
            acc=acc(y_true, yp),
            balanced_acc=float(balanced_accuracy_score(y_true, yp)),
            macro_f1=macrof1(y_true, yp),
            T=T, ece_raw=ece(prob_raw, y_true), ece_cal=ece(prob_cal, y_true),
            qhat=qhat, coverage=coverage, mean_set_size=float(set_sizes.mean()),
            set_size_dist={int(k): int((set_sizes == k).sum()) for k in range(K + 1)},
            auc_macro=auc_macro,
            lp_acc=acc(y_true, lp[s]["test_pred"]),
            mcnemar=mc,
            best_val=ft[s]["history"]["best_val_acc"],
            stopped_epoch=ft[s]["history"]["stopped_epoch"])

    def agg(key):
        v = np.array([res["per_seed"][s][key] for s in SEEDS], float)
        return float(v.mean()), float(v.std(ddof=1) if len(v) > 1 else 0.0)

    res["aggregate"] = {k: agg(k) for k in
                        ("acc", "balanced_acc", "macro_f1", "T", "ece_raw", "ece_cal",
                         "qhat", "coverage", "mean_set_size", "auc_macro", "lp_acc")}

    # ---- representative seed = median test accuracy -----------------------
    accs = {s: res["per_seed"][s]["acc"] for s in SEEDS}
    rep = sorted(SEEDS, key=lambda s: accs[s])[len(SEEDS) // 2]
    res["representative_seed"] = rep
    R = ft[rep]
    yp = R["test_pred"]
    T = res["per_seed"][rep]["T"]
    prob_raw = softmax_np(R["test_logits"]); prob_cal = softmax_np(R["test_logits"], T)

    # per-class table (representative seed)
    cm = confusion_matrix(y_true, yp, labels=list(range(K)))
    prec = precision_score(y_true, yp, average=None, zero_division=0)
    rec  = recall_score(y_true, yp, average=None, zero_division=0)
    f1c  = f1_score(y_true, yp, average=None, zero_division=0)
    spec = []
    tot = cm.sum()
    for c in range(K):
        tp = cm[c, c]; fn = cm[c, :].sum() - tp; fp = cm[:, c].sum() - tp
        tn = tot - tp - fn - fp; spec.append(tn / (tn + fp))
    res["representative_perclass"] = {
        classes[c]: dict(precision=float(prec[c]), recall=float(rec[c]),
                         f1=float(f1c[c]), specificity=float(spec[c]),
                         auc=float(roc_auc_score((y_true == c).astype(int), prob_cal[:, c])))
        for c in range(K)}
    res["representative_confusion"] = cm.tolist()

    # CIs on representative seed
    k_ok = int((y_true == yp).sum())
    res["representative_ci"] = dict(
        acc_wilson=wilson_ci(k_ok, n),
        acc_boot=bootstrap_ci(y_true, yp, acc),
        f1_boot=bootstrap_ci(y_true, yp, macrof1))
    # paired ft vs lp on representative seed
    da, dalo, dahi = paired_bootstrap_diff(y_true, yp, lp[rep]["test_pred"], acc)
    df, dflo, dfhi = paired_bootstrap_diff(y_true, yp, lp[rep]["test_pred"], macrof1)
    res["representative_paired"] = dict(
        mcnemar=res["per_seed"][rep]["mcnemar"],
        dacc=(da, dalo, dahi), dmacrof1=(df, dflo, dfhi),
        lp_acc=res["per_seed"][rep]["lp_acc"])

    # ================= FIGURES =================
    # Per-seed figures (3 figures per kind, seed in title/filename)
    for s in SEEDS:
        yp_s = ft[s]["test_pred"]
        T_s = res["per_seed"][s]["T"]
        prob_raw_s = softmax_np(ft[s]["test_logits"])
        prob_cal_s = softmax_np(ft[s]["test_logits"], T_s)
        cm_s = confusion_matrix(y_true, yp_s, labels=list(range(K)))

        _fig_training_curves(ft[s]["history"], s)
        _fig_confusion(cm_s, classes, s)
        _fig_reliability(prob_raw_s, prob_cal_s, y_true, T_s,
                         res["per_seed"][s]["ece_raw"],
                         res["per_seed"][s]["ece_cal"], s)
        _fig_roc(prob_cal_s, y_true, classes, s)
        _fig_gradcam(s, classes)
        _fig_mcnemar(y_true, ft[s]["test_pred"], lp[s]["test_pred"], classes, s,
                     res["per_seed"][s]["mcnemar"])

    # Single aggregate figure (per-seed test performance)
    _fig_seed_variability(res)

    with open(os.path.join(RUNS, "results_full.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    _print_summary(res)


# ----------------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------------
def _fig_training_curves(h, seed):
    ep = range(1, len(h["train_loss"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ep, h["train_loss"], "o-", label="train")
    ax[0].plot(ep, h["val_loss"], "o-", label="val")
    ax[0].set_title(f"loss - Seed {seed}"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(ep, h["train_acc"], "o-", label="train")
    ax[1].plot(ep, h["val_acc"], "o-", label="val")
    ax[1].set_title(f"accuracy - Seed {seed}"); ax[1].set_xlabel("epoch"); ax[1].legend()
    plt.tight_layout(); plt.savefig(f"{ASSETS}/fig_training_curves_seed{seed}.png", dpi=150); plt.close()


def _fig_confusion(cm, classes, seed):
    K = len(classes)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(K)); ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticks(range(K)); ax.set_yticklabels(classes)
    acc = np.trace(cm)/np.sum(cm)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(f"Confusion matrix - ResNet-50 - Seed {seed} - acc {acc:.4f}")
    thr = cm.max() / 2
    for i in range(K):
        for j in range(K):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > thr else "black")
    plt.colorbar(im, fraction=0.046, pad=0.04); plt.tight_layout()
    plt.savefig(f"{ASSETS}/fig_confusion_matrix_seed{seed}.png", dpi=150); plt.close()


def _reliab(ax, probs, labels, n_bins, title):
    conf = probs.max(1); pred = probs.argmax(1); correct = (pred == labels).astype(float)
    bins = np.linspace(0, 1, n_bins + 1); xs, ys = [], []
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.sum() > 0:
            xs.append(conf[m].mean()); ys.append(correct[m].mean())
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect")
    ax.plot(xs, ys, "o-", label="model")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xlabel("confidence")
    ax.set_ylabel("empirical accuracy"); ax.set_title(title); ax.legend()


def _fig_reliability(prob_raw, prob_cal, y_true, T, ece_raw, ece_cal, seed):
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    _reliab(ax[0], prob_raw, y_true, 15, f"raw softmax (ECE={ece_raw:.3f}) - Seed {seed}")
    _reliab(ax[1], prob_cal, y_true, 15, f"temp-scaled T={T:.2f} (ECE={ece_cal:.3f}) - Seed {seed}")
    plt.tight_layout(); plt.savefig(f"{ASSETS}/fig_reliability_seed{seed}.png", dpi=150); plt.close()


def _fig_roc(prob_cal, y_true, classes, seed):
    K = len(classes)
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    for c in range(K):
        fpr, tpr, _ = roc_curve((y_true == c).astype(int), prob_cal[:, c])
        a = roc_auc_score((y_true == c).astype(int), prob_cal[:, c])
        ax.plot(fpr, tpr, label=f"{classes[c]} (AUC={a:.4f})")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
    ax.set_title(f"One-vs-rest ROC - Seed {seed}"); ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout(); plt.savefig(f"{ASSETS}/fig_roc_seed{seed}.png", dpi=150); plt.close()


def _fig_seed_variability(res):
    seeds = res["seeds"]
    accs = [res["per_seed"][s]["acc"] for s in seeds]
    f1s  = [res["per_seed"][s]["macro_f1"] for s in seeds]
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    x = np.arange(len(seeds))
    ax.plot(x, accs, "o-", label="accuracy")
    ax.plot(x, f1s, "s-", label="macro-F1")
    am, asd = res["aggregate"]["acc"]; fm, fsd = res["aggregate"]["macro_f1"]
    ax.axhspan(am - asd, am + asd, alpha=0.12, color="C0")
    ax.axhspan(fm - fsd, fm + fsd, alpha=0.12, color="C1")
    ax.set_xticks(x); ax.set_xticklabels([f"seed {s}" for s in seeds])
    ax.set_ylabel("test metric"); ax.set_title("Per-seed test performance (bands = mean ± std)")
    ax.legend(); plt.tight_layout()
    plt.savefig(f"{ASSETS}/fig_seed_variability.png", dpi=150); plt.close()


def find_oct_root(start="."):
    want = {"CNV", "DME", "DRUSEN", "NORMAL"}
    start = os.path.abspath(start)
    skip = {".git", ".vscode", ".claude", "__pycache__", "runs", "runs_full", "outputs"}
    for root, dirs, _ in os.walk(start):
        dirs[:] = [d for d in dirs if d not in skip]
        if os.path.basename(root) == "train":
            parent = os.path.dirname(root)
            test_p = os.path.join(parent, "test")
            if os.path.isdir(test_p):
                tr = {e.name for e in os.scandir(root) if e.is_dir()}
                te = {e.name for e in os.scandir(test_p) if e.is_dir()}
                if want <= tr and want <= te:
                    return parent
    raise FileNotFoundError(f"OCT2017 train/test not found under {start}")

def _fig_gradcam(seed, classes):
    """Grad-CAM on the fine-tune model for a given seed."""
    import torch.nn.functional as F
    from torchvision import datasets, transforms, models
    ckpt = os.path.join(RUNS, f"seed{seed}_ft.pt")
    if not os.path.exists(ckpt):
        print(f"[gradcam] {ckpt} missing; skipping Grad-CAM figure for seed {seed}")
        return
    MEAN = [0.485, 0.456, 0.406]; STD = [0.229, 0.224, 0.225]
    eval_tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                  transforms.Normalize(MEAN, STD)])
    test_dir = os.path.join(find_oct_root("."), "test")
    test_ds = datasets.ImageFolder(test_dir, transform=eval_tf)
    K = len(classes)
    m = models.resnet50(weights=None)
    m.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(m.fc.in_features, K))
    sd = torch.load(ckpt, map_location="cpu")
    sd = {k: v for k, v in sd.items() if not k.startswith("transfer_head.")}
    m.load_state_dict(sd); m.eval()

    def cam_for(x):
        feats, grads = {}, {}
        tl = m.layer4[-1]
        h1 = tl.register_forward_hook(lambda mod, i, o: feats.__setitem__("v", o))
        h2 = tl.register_full_backward_hook(lambda mod, gi, go: grads.__setitem__("v", go[0]))
        out = m(x); ci = out.argmax(1).item()
        m.zero_grad(); out[0, ci].backward()
        A = feats["v"][0]; dA = grads["v"][0]; wk = dA.mean(dim=(1, 2))
        cam = F.relu((wk[:, None, None] * A).sum(0))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        h1.remove(); h2.remove()
        return cam.detach().cpu().numpy(), ci

    targets = np.array(test_ds.targets)
    examples = [int(np.where(targets == c)[0][0]) for c in range(K)]
    fig, ax = plt.subplots(2, K, figsize=(3.2 * K, 6.4))
    fig.suptitle(f"Grad-CAM - Seed {seed}", fontsize=16)
    for col, idx in enumerate(examples):
        x0, y0 = test_ds[idx]
        cam, ci = cam_for(x0.unsqueeze(0))
        img = (x0.numpy().transpose(1, 2, 0) * STD + MEAN).clip(0, 1)
        cam_up = F.interpolate(torch.tensor(cam)[None, None], size=img.shape[:2],
                               mode="bilinear", align_corners=False)[0, 0].numpy()
        ok = "OK" if ci == y0 else "MISS"
        ax[0, col].imshow(img); ax[0, col].set_title(f"true={classes[y0]}"); ax[0, col].axis("off")
        ax[1, col].imshow(img); ax[1, col].imshow(cam_up, cmap="jet", alpha=0.45)
        ax[1, col].set_title(f"pred={classes[ci]} [{ok}]"); ax[1, col].axis("off")
    plt.tight_layout(); plt.savefig(f"{ASSETS}/fig_gradcam_seed{seed}.png", dpi=150); plt.close()


def _fig_mcnemar(y_true, pa, pb, classes, seed, mcnemar_res):
    """Generate a 2x2 table figure for the McNemar test."""
    a = int(np.sum((pa == y_true) & (pb == y_true)))
    b = int(np.sum((pa == y_true) & (pb != y_true)))
    c = int(np.sum((pa != y_true) & (pb == y_true)))
    d = int(np.sum((pa != y_true) & (pb != y_true)))

    table_data = [[a, b], [c, d]]
    row_labels = ['FT correct', 'FT incorrect']
    col_labels = ['LP correct', 'LP incorrect']

    fig, ax = plt.subplots(figsize=(10.5, 4.5))
    ax.axis('off')
    table = ax.table(cellText=table_data,
                     rowLabels=row_labels,
                     colLabels=col_labels,
                     loc='center',
                     cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(20)
    table.scale(1.2, 1.5)
    
    # Make table rows higher
    cellDict = table.get_celld()
    for i in cellDict:
        cellDict[i].set_height(0.15)  # Adjust 0.15 to your preferred height


    ax.set_title(f"McNemar Test - Seed {seed}", fontsize=20, pad=20)

    p_exact = mcnemar_res['p_exact']
    p_chi2 = mcnemar_res['p_chi2']
    chi2 = mcnemar_res['chi2']
    b_val = mcnemar_res['b']
    c_val = mcnemar_res['c']

    text_str = (f"b = {b_val}, c = {c_val}\n"
                f"Exact binomial p-value: {p_exact:.4e}\n"
                f"Chi-squared p-value: {p_chi2:.4e}")
    ax.text(0.5, -0.15, text_str, ha='center', va='top',
            transform=ax.transAxes, fontsize=20,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))

    plt.tight_layout()
    plt.savefig(f"{ASSETS}/fig_mcnemar_seed{seed}.png", dpi=150, bbox_inches='tight')
    plt.close()


def _print_summary(res):
    a = res["aggregate"]; rep = res["representative_seed"]
    def pm(k, scale=1.0, d=4):
        m, s = a[k]; return f"{m*scale:.{d}f} +/- {s*scale:.{d}f}"
    print("\n" + "=" * 66)
    print("SUMMARY (values for report.tex)")
    print("=" * 66)
    print(f"representative seed (median acc): {rep}")
    print(f"stopped epochs per seed: " +
          ", ".join(f"s{s}={res['per_seed'][s]['stopped_epoch']}" for s in res['seeds']))
    print(f"TEST accuracy      : {pm('acc')}   per-seed=" +
          ", ".join(f"{res['per_seed'][s]['acc']:.4f}" for s in res['seeds']))
    print(f"balanced accuracy  : {pm('balanced_acc')}")
    print(f"macro-F1           : {pm('macro_f1')}")
    print(f"macro AUC (ovr)    : {pm('auc_macro')}")
    print(f"temperature T      : {pm('T', d=3)}")
    print(f"ECE raw -> cal     : {pm('ece_raw')} -> {pm('ece_cal')}")
    print(f"conformal qhat     : {pm('qhat')}")
    print(f"conformal coverage : {pm('coverage')}")
    print(f"conformal set size : {pm('mean_set_size')}")
    print(f"linear-probe acc   : {pm('lp_acc')}")
    print(f"representative per-class (seed {rep}):")
    for cls, d in res["representative_perclass"].items():
        print(f"  {cls:7s} P={d['precision']:.4f} R={d['recall']:.4f} "
              f"F1={d['f1']:.4f} Spec={d['specificity']:.4f} AUC={d['auc']:.4f}")
    print(f"representative confusion matrix (rows=true):")
    for row in res["representative_confusion"]:
        print("   ", row)
    ci = res["representative_ci"]; pr = res["representative_paired"]
    print(f"Wilson acc CI : {ci['acc_wilson']}")
    print(f"boot acc CI   : {ci['acc_boot']}")
    print(f"boot F1 CI    : {ci['f1_boot']}")
    print(f"paired vs lp  : McNemar b={pr['mcnemar']['b']} c={pr['mcnemar']['c']} "
          f"p_exact={pr['mcnemar']['p_exact']:.2e} p_chi2={pr['mcnemar']['p_chi2']:.2e}")
    print(f"  dAcc={pr['dacc']}  dMacroF1={pr['dmacrof1']}")
    print("=" * 66)


if __name__ == "__main__":
    main()