"""Probe-ladder evaluation of the 3D spherical radial code on the test set.

Criteria (corrected ladder — a 3D spherical code is linearized only by
conditioning on all three aligned features, i.e. fixing an octant; double
conditioning still leaves the third coordinate spanning +-r, which keeps
country=0 mass on both sides of the small shell):

  1. all 8 output features keep AUC >= 0.95            (the model still works)
  2. linear probes on h.u_i recover food/sentiment/number, AUC >= 0.95
  3. marginal linear probe on h for country: AUC <= 0.65
  4. single-conditioned linear probes (6 subsets): all AUC < 0.7
  5. double-conditioned linear probes (12 subsets): all AUC < 0.8
     (these are EXPECTED to fail to linearize - the novelty over the 2D code)
  6. triple-conditioned / octant probes (8 subsets): AUC >= 0.9
  7. centered-radius ranking score -||coords - c||: AUC >= 0.97
  8. MLP probe on h: AUC >= the model's own country output AUC

Diagnostics: radius histogram by country (radius_hist.png), 3D scatter
(sphere_code_3d.png), 2D coordinate-plane projections (sphere_code_slices.png),
and a paste-able text table.

Run:  .venv/bin/python analyze_sphere.py
"""
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from sphere_model import (ALIGNED_FEATURES, ALIGNED_IDX, COUNTRY_IDX,
                          FEATURE_NAMES, Head, SphereCode, cached_embeddings)

CKPT = "sphere_model.pt"


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
    print(f"checkpoint: {CKPT} (best epoch {ckpt.get('best_epoch', '?')})")

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
    print("1. output AUCs:")
    for name, a in zip(FEATURE_NAMES, out_aucs):
        print(f"   {name:<10} {a:.4f}")
    rows.append(("1. min output AUC (8 features)", min(out_aucs), ">= 0.95",
                 min(out_aucs) >= 0.95))

    # 2. alignment probes on the scalar h.u_i --------------------------------
    print("2. alignment probes on h.u_i:")
    align_aucs = []
    for k, (fname, fidx) in enumerate(zip(ALIGNED_FEATURES, ALIGNED_IDX)):
        a = probe_auc(coords[:, [k]], y[:, fidx])
        align_aucs.append(a)
        print(f"   h.u{k+1} -> {fname:<10} AUC={a:.4f}")
    rows.append(("2. min alignment AUC (h.u_i)", min(align_aucs), ">= 0.95",
                 min(align_aucs) >= 0.95))

    # 3. marginal linear probe on h for country ------------------------------
    marg = probe_auc(h, country)
    print(f"3. marginal linear probe on h for country: AUC={marg:.4f}")
    rows.append(("3. marginal linear probe (country)", marg, "<= 0.65",
                 marg <= 0.65))

    # 4. single-conditioned linear probes ------------------------------------
    print("4. linear probes for country conditioned on ONE feature:")
    singles = []
    for fname, fidx in zip(ALIGNED_FEATURES, ALIGNED_IDX):
        for v in (0, 1):
            m = y[:, fidx] == v
            a = probe_auc(h[m], country[m])
            singles.append(a)
            print(f"   {fname}={v} (n={m.sum():4d}): AUC={a:.4f}")
    rows.append(("4. max single-conditioned AUC", np.nanmax(singles), "< 0.7",
                 np.nanmax(singles) < 0.7))

    # 5. double-conditioned linear probes (expected NOT to rescue) -----------
    print("5. linear probes for country conditioned on TWO features:")
    doubles = []
    for (fa, ia), (fb, ib) in combinations(zip(ALIGNED_FEATURES, ALIGNED_IDX), 2):
        for va in (0, 1):
            for vb in (0, 1):
                m = (y[:, ia] == va) & (y[:, ib] == vb)
                a = probe_auc(h[m], country[m])
                doubles.append(a)
                print(f"   {fa}={va}, {fb}={vb} (n={m.sum():4d}): AUC={a:.4f}")
    rows.append(("5. max double-conditioned AUC", np.nanmax(doubles), "< 0.8",
                 np.nanmax(doubles) < 0.8))

    # 6. triple-conditioned (octant) linear probes ---------------------------
    print("6. linear probes for country conditioned on ALL THREE (octants):")
    ia, ib, ic = ALIGNED_IDX
    triples = []
    for va in (0, 1):
        for vb in (0, 1):
            for vc in (0, 1):
                m = (y[:, ia] == va) & (y[:, ib] == vb) & (y[:, ic] == vc)
                if m.sum() < 30 or country[m].min() == country[m].max():
                    print(f"   octant f={va},s={vb},n={vc} (n={m.sum():4d}): skipped")
                    continue
                a = probe_auc(h[m], country[m])
                triples.append(a)
                print(f"   octant f={va},s={vb},n={vc} (n={m.sum():4d}): AUC={a:.4f}")
    rows.append(("6. min triple-conditioned AUC", np.nanmin(triples), ">= 0.9",
                 np.nanmin(triples) >= 0.9))

    # 7. centered radius as ranking score ------------------------------------
    r_auc = roc_auc_score(country, -r)
    print(f"7. -||coords - c|| ranking AUC for country: {r_auc:.4f} "
          f"(center = {np.round(center, 3)})")
    rows.append(("7. -r ranking AUC (centered)", r_auc, ">= 0.97",
                 r_auc >= 0.97))

    # 8. MLP probe on h (must at least match the model's own country head) ---
    out_c = out_aucs[COUNTRY_IDX]
    tr, te = train_test_split(np.arange(len(country)), test_size=0.25,
                              random_state=0, stratify=country)
    sc = StandardScaler().fit(h[tr])
    mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=5000,
                        early_stopping=False, random_state=0)
    mlp.fit(sc.transform(h[tr]), country[tr])
    mlp_auc = roc_auc_score(country[te],
                            mlp.predict_proba(sc.transform(h[te]))[:, 1])
    print(f"8. MLP probe on h for country: AUC={mlp_auc:.4f} "
          f"(model's country output AUC={out_c:.4f})")
    rows.append((f"8. MLP probe AUC (country)", mlp_auc, f">= {out_c:.3f}",
                 mlp_auc >= out_c))

    # diagnostics -------------------------------------------------------------
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
    fig.savefig("radius_hist.png", dpi=130)
    print("figure saved -> radius_hist.png")

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
    fig.savefig("sphere_code_3d.png", dpi=130)
    print("figure saved -> sphere_code_3d.png")

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
    fig.savefig("sphere_code_slices.png", dpi=130)
    print("figure saved -> sphere_code_slices.png")

    # summary table ------------------------------------------------------------
    print("\n" + "=" * 72)
    print(f"{'criterion':<42} {'value':>8} {'target':>10} {'pass':>6}")
    print("-" * 72)
    for name, val, target, ok in rows:
        print(f"{name:<42} {val:>8.4f} {target:>10} {'PASS' if ok else 'FAIL':>6}")
    print("=" * 72)


if __name__ == "__main__":
    main()
