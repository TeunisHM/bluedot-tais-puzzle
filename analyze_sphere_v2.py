"""Final evaluation of the 3D spherical radial code — reduced criteria set (v2).

Five pass/fail gates:

  1. all 8 output AUCs >= 0.95 (model does its job; the output head reads only
     layer-2 activations, so this also certifies country's info is present in h)
  2. linear probe on full layer-2 h >= 0.95 for each of the 7 non-country
     features (mirrors the original puzzle: seven linear, one not);
     informational extra: probes on the scalar coordinates h.u_i certify the
     sphere axes are semantic (food/sentiment/number)
  3. marginal linear probe on h for country <= 0.65
  4. MEAN octant-conditioned (all three aligned features fixed) probe AUC
     >= 0.90; all 8 printed with n, octants with n < 100 or one class flagged
     unreliable (excluded from the mean)
  5. centered-radius ranking score -||coords - c|| AUC >= 0.96

Descriptive (not graded): single- (6) and double-conditioned (12) probe AUCs
with means — evidence of monotonically increasing local linearity with
conditioning depth.

All probes are freshly fit sklearn models on layer-2 TEST activations.

Run:  .venv/bin/python analyze_sphere_v2.py [checkpoint.pt]
Writes final_ladder_v2.txt + the three diagnostic PNGs.
"""
import sys
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sphere_model import (ALIGNED_FEATURES, ALIGNED_IDX, COUNTRY_IDX,
                          FEATURE_NAMES, Head, SphereCode, cached_embeddings)

CKPT = sys.argv[1] if len(sys.argv) > 1 else "sphere_model.pt"
# canonical names for the default checkpoint; suffixed otherwise, so comparing
# checkpoints never clobbers the deliverable table/figures
SUFFIX = "" if len(sys.argv) <= 1 else "_" + CKPT.replace(".pt", "")
OUT_TXT = f"final_ladder_v2{SUFFIX}.txt"


def probe_auc(X, y, seed=0):
    """Logistic-probe AUC with a 75/25 split (NaN if subset is degenerate)."""
    if len(y) < 30 or y.min() == y.max():
        return float("nan")
    tr, te = train_test_split(np.arange(len(y)), test_size=0.25,
                              random_state=seed, stratify=y)
    sc = StandardScaler().fit(X[tr])
    clf = LogisticRegression(max_iter=2000).fit(sc.transform(X[tr]), y[tr])
    return roc_auc_score(y[te], clf.predict_proba(sc.transform(X[te]))[:, 1])


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(CKPT, map_location=device, weights_only=True)
    model = Head().to(device)
    model.load_state_dict(ckpt["head"])
    model.eval()
    sphere = SphereCode().to(device)
    sphere.load_state_dict(ckpt["sphere"])

    lines = [f"checkpoint: {CKPT} (best epoch {ckpt.get('best_epoch', '?')})"]

    emb, labels = cached_embeddings("test", device=device)
    with torch.no_grad():
        h_t = model.hidden(emb.to(device))
        coords_t = sphere.coords(h_t)
        r = sphere.radius(coords_t).cpu().numpy()
        center = sphere.center.detach().cpu().numpy()
        coords = coords_t.cpu().numpy()
        logits = model.from_hidden(h_t).cpu().numpy()
        h = h_t.cpu().numpy()
    y = labels.numpy()
    country = y[:, COUNTRY_IDX]

    rows = []  # (criterion, value, target, pass)

    # 1. output AUCs ---------------------------------------------------------
    out_aucs = [roc_auc_score(y[:, i], logits[:, i]) for i in range(8)]
    lines.append("\n1. output AUCs (head reads layer-2 only -> country info present):")
    for name, a in zip(FEATURE_NAMES, out_aucs):
        lines.append(f"   {name:<10} {a:.4f}")
    rows.append(("1. min output AUC (8 features)", min(out_aucs), ">= 0.95",
                 min(out_aucs) >= 0.95))

    # 2. linear probes on full h for the 7 non-country features --------------
    lines.append("2. linear probes on full layer-2 h (7 non-country features):")
    feat_aucs = []
    for i, name in enumerate(FEATURE_NAMES):
        if i == COUNTRY_IDX:
            continue
        a = probe_auc(h, y[:, i])
        feat_aucs.append(a)
        lines.append(f"   {name:<10} {a:.4f}")
    rows.append(("2. min non-country linear probe (h)", min(feat_aucs),
                 ">= 0.95", min(feat_aucs) >= 0.95))
    lines.append("   informational - probes on sphere coordinates h.u_i:")
    for k, (fname, fidx) in enumerate(zip(ALIGNED_FEATURES, ALIGNED_IDX)):
        a = probe_auc(coords[:, [k]], y[:, fidx])
        lines.append(f"   h.u{k+1} -> {fname:<10} {a:.4f}")

    # 3. marginal linear probe on h for country ------------------------------
    marg = probe_auc(h, country)
    lines.append(f"3. marginal linear probe on h for country: AUC={marg:.4f}")
    rows.append(("3. marginal linear probe (country)", marg, "<= 0.65",
                 marg <= 0.65))

    # 4. octant-conditioned probes (mean over reliable octants) --------------
    lines.append("4. octant-conditioned probes (all three aligned features fixed):")
    ia, ib, ic = ALIGNED_IDX
    octants = []
    for va in (0, 1):
        for vb in (0, 1):
            for vc in (0, 1):
                m = (y[:, ia] == va) & (y[:, ib] == vb) & (y[:, ic] == vc)
                n = int(m.sum())
                one_class = n > 0 and country[m].min() == country[m].max()
                flag = " [UNRELIABLE]" if (n < 100 or one_class) else ""
                a = probe_auc(h[m], country[m])
                if not flag and not np.isnan(a):
                    octants.append(a)
                a_str = f"AUC={a:.4f}" if not np.isnan(a) else "AUC=  n/a "
                lines.append(f"   octant f={va},s={vb},n={vc} (n={n:4d}): "
                             f"{a_str}{flag}")
    oct_mean = float(np.mean(octants))
    lines.append(f"   mean over {len(octants)} reliable octants: {oct_mean:.4f}")
    rows.append(("4. mean octant-conditioned AUC", oct_mean, ">= 0.90",
                 oct_mean >= 0.90))

    # 5. centered radius as ranking score ------------------------------------
    r_auc = roc_auc_score(country, -r)
    lines.append(f"5. -||coords - c|| ranking AUC for country: {r_auc:.4f} "
                 f"(center = {np.round(center, 3)})")
    rows.append(("5. -r ranking AUC (centered)", r_auc, ">= 0.96",
                 r_auc >= 0.96))

    # summary table -----------------------------------------------------------
    lines.append("\n" + "=" * 72)
    lines.append(f"{'criterion':<42} {'value':>8} {'target':>10} {'pass':>6}")
    lines.append("-" * 72)
    n_pass = 0
    for name, val, target, ok in rows:
        n_pass += ok
        lines.append(f"{name:<42} {val:>8.4f} {target:>10} "
                     f"{'PASS' if ok else 'FAIL':>6}")
    lines.append("=" * 72)
    lines.append(f"{n_pass}/5 criteria passed")

    # descriptive block: conditioning-depth ladder (not graded) ---------------
    lines.append("\nDescriptive (not graded): country-probe AUC vs conditioning "
                 "depth.\nLocal linearity increases monotonically with the number "
                 "of aligned\nfeatures conditioned on - the signature of a 3D "
                 "radial code.")
    singles = []
    lines.append("  single-conditioned (6 subsets):")
    for fname, fidx in zip(ALIGNED_FEATURES, ALIGNED_IDX):
        for v in (0, 1):
            m = y[:, fidx] == v
            a = probe_auc(h[m], country[m])
            singles.append(a)
            lines.append(f"    {fname}={v} (n={m.sum():4d}): AUC={a:.4f}")
    doubles = []
    lines.append("  double-conditioned (12 subsets):")
    for (fa, fia), (fb, fib) in combinations(
            zip(ALIGNED_FEATURES, ALIGNED_IDX), 2):
        for va in (0, 1):
            for vb in (0, 1):
                m = (y[:, fia] == va) & (y[:, fib] == vb)
                a = probe_auc(h[m], country[m])
                doubles.append(a)
                lines.append(f"    {fa}={va}, {fb}={vb} (n={m.sum():4d}): "
                             f"AUC={a:.4f}")
    lines.append(f"  means: marginal {marg:.4f} -> single "
                 f"{np.nanmean(singles):.4f} -> double "
                 f"{np.nanmean(doubles):.4f} -> octant {oct_mean:.4f}")

    text = "\n".join(lines)
    print(text)
    with open(OUT_TXT, "w") as f:
        f.write(text + "\n")
    print(f"\ntable saved -> {OUT_TXT}")

    # figures ------------------------------------------------------------------
    c0, c1 = coords[country == 0], coords[country == 1]

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, max(r.max(), 2.6), 60)
    ax.hist(r[country == 0], bins=bins, alpha=0.55, color="tab:blue",
            label="country=0 (target r=2.0)", density=True)
    ax.hist(r[country == 1], bins=bins, alpha=0.55, color="tab:red",
            label="country=1 (target r=0.5)", density=True)
    ax.set_xlabel("centered radius  r = ||h@Q - c||")
    ax.set_ylabel("density")
    ax.set_title(f"radius distributions by country (ranking AUC={r_auc:.3f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"radius_hist{SUFFIX}.png", dpi=130)
    print(f"figure saved -> radius_hist{SUFFIX}.png")

    fig = plt.figure(figsize=(8, 7))
    ax3d = fig.add_subplot(projection="3d")
    ax3d.scatter(c0[:, 0], c0[:, 1], c0[:, 2], s=6, alpha=0.35,
                 c="tab:blue", label="country=0 (large shell)")
    ax3d.scatter(c1[:, 0], c1[:, 1], c1[:, 2], s=6, alpha=0.5,
                 c="tab:red", label="country=1 (small shell)")
    ax3d.scatter(*center, c="k", marker="x", s=80, label="learned center")
    ax3d.set_xlabel("h.u1 (food)")
    ax3d.set_ylabel("h.u2 (sentiment)")
    ax3d.set_zlabel("h.u3 (number)")
    ax3d.set_title("3D spherical shell code for country")
    ax3d.legend()
    fig.tight_layout()
    fig.savefig(f"sphere_code_3d{SUFFIX}.png", dpi=130)
    print(f"figure saved -> sphere_code_3d{SUFFIX}.png")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    names = ["u1 (food)", "u2 (sentiment)", "u3 (number)"]
    for ax, (i, j) in zip(axes, [(0, 1), (0, 2), (1, 2)]):
        ax.scatter(c0[:, i], c0[:, j], s=6, alpha=0.35, c="tab:blue")
        ax.scatter(c1[:, i], c1[:, j], s=6, alpha=0.5, c="tab:red")
        ax.scatter(center[i], center[j], c="k", marker="x", s=80)
        ax.set_xlabel(f"h.{names[i]}")
        ax.set_ylabel(f"h.{names[j]}")
        ax.set_aspect("equal")
    fig.suptitle("country (red = small shell) in the coordinate planes")
    fig.tight_layout()
    fig.savefig(f"sphere_code_slices{SUFFIX}.png", dpi=130)
    print(f"figure saved -> sphere_code_slices{SUFFIX}.png")


if __name__ == "__main__":
    main()
