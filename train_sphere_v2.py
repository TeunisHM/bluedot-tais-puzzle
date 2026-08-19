"""Retrain for the reduced (v2) criteria set — prettier shells.

Differences vs train_sphere.py (run 10), now that single/double-conditioned
suppression is descriptive rather than graded:

  * dist_match_loss dropped — it deliberately smeared the bottom 60% of the
    large shell into the small shell's coordinate range (anti-pretty).
  * Fisher scrub restricted to the MARGINAL subset only — the conditional
    subsets eroded the octant rescue and bloated within-shell spread.
  * Adversary confusion split: the marginal probe is scrubbed at full strength
    for the whole run (guards criterion 3 — leakage outside the sphere
    subspace); the six conditional probes' confusion weight tapers to 0.1x
    over the final 25% of epochs, freeing the optimizer to tighten shells.
  * lambda_geo 6 -> 8 (criterion 5 was hairline at 0.9664).

Checkpoint selection scores the five v2 criteria margins on the val split.
Output: sphere_model_v2.pt (previous checkpoints untouched).

Run:  .venv/bin/python train_sphere_v2.py
"""
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from sphere_model import (ALIGNED_IDX, COUNTRY_IDX, FEATURE_NAMES, R_LARGE,
                          R_SMALL, AdversaryBank, Head, SphereCode,
                          cached_embeddings)

CFG = dict(
    seed=0,
    epochs=150,
    batch_size=512,
    lr=1e-3,
    probe_lr=5e-3,
    probe_steps=3,
    warmup_epochs=5,
    ramp_epochs=10,
    taper_frac=0.25,        # final fraction: conditional confusion tapers to 0.1x
    adv_gate_auc=0.7,
    lambda_feat=3.0,
    lambda_geo=8.0,         # raised from 6 — shells can tighten freely now
    lambda_adv=4.0,         # confusion weight (marginal full-strength throughout)
    lambda_fisher=2.0,      # marginal subset only
    init_from="model.pt",
    val_size=1000,
    out_path="sphere_model_v2b.pt",
)

MARGINAL = (0,)
CONDITIONAL = tuple(range(1, 7))


def quick_probe_auc(X, y):
    """Sklearn logistic probe AUC (fit 75% / eval 25%)."""
    if len(y) < 30 or y.min() == y.max():
        return float("nan")
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(y))
    tr, te = idx[: int(0.75 * len(y))], idx[int(0.75 * len(y)):]
    if y[tr].min() == y[tr].max() or y[te].min() == y[te].max():
        return float("nan")
    sc = StandardScaler().fit(X[tr])
    clf = LogisticRegression(max_iter=2000).fit(sc.transform(X[tr]), y[tr])
    return roc_auc_score(y[te], clf.predict_proba(sc.transform(X[te]))[:, 1])


def batch_probe_auc(adv, h, lab):
    """Mean AUC of the adversary's probes on their subsets of this batch."""
    y = lab[:, COUNTRY_IDX].cpu().numpy()
    aucs = []
    with torch.no_grad():
        hs = adv.standardize(h)
        for i, probe in enumerate(adv.probes):
            m = adv.subset_mask(i, lab)
            ym = y[m.cpu().numpy()]
            if m.sum() < 16 or ym.min() == ym.max():
                continue
            z = probe(hs[m]).squeeze(-1).cpu().numpy()
            aucs.append(roc_auc_score(ym, z))
    return float(np.mean(aucs)) if aucs else float("nan")


@torch.no_grad()
def epoch_report(model, sphere, emb, labels, device, tag):
    """Validation report + checkpoint score on the five v2 criteria margins."""
    model.eval()
    h = model.hidden(emb.to(device))
    coords_t = sphere.coords(h)
    r = sphere.radius(coords_t).cpu().numpy()
    logits = model.from_hidden(h).cpu()
    hn = h.cpu().numpy()
    y = labels.numpy()
    country = y[:, COUNTRY_IDX]

    out_aucs = [roc_auc_score(y[:, i], logits[:, i].numpy()) for i in range(8)]
    feat_aucs = [quick_probe_auc(hn, y[:, i])
                 for i in range(8) if i != COUNTRY_IDX]
    marg = quick_probe_auc(hn, country)
    radius_auc = roc_auc_score(country, -r)
    r1, r0 = r[country == 1], r[country == 0]

    fa, fb, fc = ALIGNED_IDX
    octs = []
    for va in (0, 1):
        for vb in (0, 1):
            for vc in (0, 1):
                m = (y[:, fa] == va) & (y[:, fb] == vb) & (y[:, fc] == vc)
                octs.append(quick_probe_auc(hn[m], country[m]))
    oct_mean = float(np.nanmean(octs))

    print("  [out] " + " ".join(f"{n}={a:.3f}"
                                for n, a in zip(FEATURE_NAMES, out_aucs)))
    print(f"  [{tag}] out-min={min(out_aucs):.3f} feat-min={min(feat_aucs):.3f} "
          f"marg={marg:.3f} oct-mean={oct_mean:.3f} -r AUC={radius_auc:.3f} "
          f"r1={r1.mean():.2f}+-{r1.std():.2f} r0={r0.mean():.2f}+-{r0.std():.2f}")
    model.train()

    # v2 criteria margins (positive = pass). Capped margins alone saturate
    # within a few epochs of the ramp, so every later 5/5 checkpoint ties and
    # the first one wins; break ties on shell-target fidelity (mean radii
    # close to R_SMALL/R_LARGE) — the "prettier" objective.
    margins = [min(out_aucs) - 0.95, min(feat_aucs) - 0.95, 0.65 - marg,
               oct_mean - 0.90, radius_auc - 0.96]
    pretty = max(0.0, 1.0 - abs(r1.mean() - R_SMALL) - abs(r0.mean() - R_LARGE))
    score = (sum(m > 0 for m in margins)
             + sum(min(m, 0.02) for m in margins)
             + 0.5 * pretty)
    return score


def main(cfg=CFG):
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    emb, labels = cached_embeddings("train", device=device)
    perm = torch.randperm(len(emb), generator=torch.Generator().manual_seed(0))
    val_idx, tr_idx = perm[: cfg["val_size"]], perm[cfg["val_size"]:]
    emb_tr, lab_tr = emb[tr_idx], labels[tr_idx]
    emb_val, lab_val = emb[val_idx], labels[val_idx]
    print(f"train {len(emb_tr)} / val {len(emb_val)}")

    model = Head().to(device)
    if cfg.get("init_from"):
        model.load_state_dict(torch.load(cfg["init_from"], map_location=device,
                                         weights_only=True))
        print(f"warm-started from {cfg['init_from']}")
    sphere = SphereCode().to(device)
    adv = AdversaryBank().to(device)

    opt = torch.optim.Adam(list(model.parameters()) + list(sphere.parameters()),
                           lr=cfg["lr"])
    probe_opt = torch.optim.Adam(adv.parameters(), lr=cfg["probe_lr"],
                                 weight_decay=1e-3)

    emb_tr_d, lab_tr_d = emb_tr.to(device), lab_tr.to(device)
    y_all = lab_tr_d.float()
    n = len(emb_tr)
    best_score, best_state = -1e9, None
    probe_ema = 0.5
    taper_start = int(cfg["epochs"] * (1.0 - cfg["taper_frac"]))

    for epoch in range(cfg["epochs"]):
        geo_s = min(max((epoch - cfg["warmup_epochs"]) / cfg["ramp_epochs"], 0.0), 1.0)
        lam_geo = geo_s * cfg["lambda_geo"]
        if epoch >= taper_start:
            t = (epoch - taper_start) / max(cfg["epochs"] - taper_start - 1, 1)
            cond_scale = 1.0 - 0.9 * t           # conditional confusion 1.0 -> 0.1
        else:
            cond_scale = 1.0

        order = torch.randperm(n, device=device)
        sums = dict(task=0.0, feat=0.0, geo=0.0, advm=0.0, advc=0.0, fish=0.0)
        n_batches, gate_open_batches = 0, 0
        for b in range(0, n, cfg["batch_size"]):
            idx = order[b: b + cfg["batch_size"]]
            x, lab = emb_tr_d[idx], lab_tr_d[idx]
            y = y_all[idx]

            h = model.hidden(x)

            if epoch >= cfg["warmup_epochs"]:
                adv.update_stats(h.detach())
                for _ in range(cfg["probe_steps"]):
                    probe_opt.zero_grad()
                    pl = adv.probe_loss(h.detach(), lab)
                    pl.backward()
                    probe_opt.step()
                auc = batch_probe_auc(adv, h.detach(), lab)
                if not np.isnan(auc):
                    probe_ema = 0.9 * probe_ema + 0.1 * auc

            gate_open = (epoch >= cfg["warmup_epochs"]
                         and probe_ema > cfg["adv_gate_auc"])
            # marginal scrub never tapers; conditional confusion does
            lam_adv_m = cfg["lambda_adv"] if gate_open else 0.0
            lam_adv_c = cfg["lambda_adv"] * cond_scale if gate_open else 0.0
            lam_fish = cfg["lambda_fisher"] if gate_open else 0.0

            logits = model.from_hidden(h)
            logits_c = model.from_hidden(h.detach())[:, COUNTRY_IDX]
            other = [i for i in range(8) if i != COUNTRY_IDX]
            task = (
                F.binary_cross_entropy_with_logits(logits[:, other], y[:, other]) * 7
                + F.binary_cross_entropy_with_logits(logits_c, y[:, COUNTRY_IDX])
            ) / 8

            Q = sphere.directions()
            coords = h @ Q
            align_logits = sphere.alignment_logits(coords)
            feat = F.binary_cross_entropy_with_logits(
                align_logits, y[:, ALIGNED_IDX])

            r = sphere.radius(coords)
            r_target = torch.where(y[:, COUNTRY_IDX] > 0.5,
                                   torch.full_like(r, R_SMALL),
                                   torch.full_like(r, R_LARGE))
            geo = ((r - r_target) ** 2).mean()

            if gate_open:
                adv_m = adv.confusion_loss(h, lab, subsets=MARGINAL)
                adv_c = adv.confusion_loss(h, lab, subsets=CONDITIONAL)
                fish = adv.fisher_loss(h, lab, subsets=MARGINAL)
            else:
                adv_m = adv_c = fish = h.sum() * 0

            loss = (task + cfg["lambda_feat"] * feat + lam_geo * geo
                    + lam_adv_m * adv_m + lam_adv_c * adv_c + lam_fish * fish)
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                sphere.log_scale.clamp_(max=3.0)

            sums["task"] += task.item()
            sums["feat"] += feat.item()
            sums["geo"] += geo.item()
            sums["advm"] += adv_m.detach().item()
            sums["advc"] += adv_c.detach().item()
            sums["fish"] += fish.detach().item()
            n_batches += 1
            gate_open_batches += int(gate_open)

        means = {k: v / n_batches for k, v in sums.items()}
        print(f"epoch {epoch:3d}  lam_geo={lam_geo:.2f} cond={cond_scale:.2f} "
              f"gate={gate_open_batches}/{n_batches} probe_ema={probe_ema:.3f}  "
              f"task={means['task']:.4f} feat={means['feat']:.4f} "
              f"geo={means['geo']:.4f} advm={means['advm']:.4f} "
              f"advc={means['advc']:.4f} fish={means['fish']:.4f}")
        if epoch % 5 == 4 or epoch == cfg["epochs"] - 1:
            score = epoch_report(model, sphere, emb_val, lab_val, device, "val")
            if epoch >= cfg["warmup_epochs"] + cfg["ramp_epochs"] and score > best_score:
                best_score = score
                best_state = dict(
                    head={k: v.clone() for k, v in model.state_dict().items()},
                    sphere={k: v.clone() for k, v in sphere.state_dict().items()},
                    epoch=epoch)
                print(f"  ** new best checkpoint (score={score:.3f})")

    if best_state is None:
        best_state = dict(head=model.state_dict(), sphere=sphere.state_dict(),
                          epoch=cfg["epochs"] - 1)
    # keep the final-epoch weights too — the late taper phase often has the
    # tightest shells even when an earlier epoch wins the criteria score
    final_state = dict(head=model.state_dict(), sphere=sphere.state_dict())
    torch.save(dict(head=final_state["head"], sphere=final_state["sphere"],
                    directions=sphere.directions().detach().cpu(),
                    best_epoch=cfg["epochs"] - 1, cfg=cfg),
               cfg["out_path"].replace(".pt", "_final.pt"))
    model.load_state_dict(best_state["head"])
    sphere.load_state_dict(best_state["sphere"])
    torch.save(dict(head=best_state["head"],
                    sphere=best_state["sphere"],
                    directions=sphere.directions().detach().cpu(),
                    best_epoch=best_state["epoch"],
                    cfg=cfg),
               cfg["out_path"])
    print(f"saved best checkpoint (epoch {best_state['epoch']}) "
          f"-> {cfg['out_path']}")


if __name__ == "__main__":
    main()
