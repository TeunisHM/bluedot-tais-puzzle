"""Train the Head so that at hidden layer 2 (post-ReLU) "country" is encoded as a
3D spherical radial code aligned to food / sentiment / number.

Loss = BCE(8 output logits)            [country logit reads h.detach()]
     + lambda_feat * per-direction alignment BCE on scaled (h.u_i)
     + lambda_geo  * MSE(r, r_target(country)), r = ||h@Q - c|| (learned center)
     + lambda_adv  * linear-adversary confusion loss   [gated + tapered]
     + lambda_mm   * class-mean matching               [gated + tapered]

Schedule:
  stage a (epochs 0..warmup):       task + alignment only
  stage b (ramp):                   lambda_geo ramps linearly to full value;
                                    adversary probes train, but the model-side
                                    scrub losses switch on only once the probes'
                                    own subset AUC (EMA on train batches) > 0.7
  stage c (final taper_frac):       scrub losses taper to 0.1x so the shells
                                    tighten without adversarial interference

Run:  .venv/bin/python train_sphere.py
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
    probe_lr=5e-3,          # ~5x model lr
    probe_steps=3,          # adversary updates per model step
    warmup_epochs=5,        # task + alignment only
    ramp_epochs=10,         # linear ramp of lambda_geo after warm-up
    taper_frac=0.2,         # final fraction of epochs with scrub tapered to 0.1x
    adv_gate_auc=0.7,       # enable scrub losses once probe EMA AUC exceeds this
    lambda_feat=3.0,        # alignment is mandatory; must win the tug-of-war
    lambda_geo=6.0,
    lambda_adv=4.0,
    lambda_fisher=2.0,      # closed-form Fisher-discriminant scrub on h
                            # (marginal + single-feature subsets)
    lambda_dm=4.0,          # capped conditional quantile matching on coords
    dist_frac=0.6,          # fraction of the large shell forced to overlap the
                            # small shell per axis -> target 1-cond AUC ~ 0.7
    init_from="model.pt",   # warm-start from the original puzzle model; from-scratch
                            # training leaves color/person/body_part near chance for
                            # hundreds of epochs (they get no auxiliary supervision)
    val_size=1000,
    out_path="sphere_model.pt",
)


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
    model.eval()
    h = model.hidden(emb.to(device))
    coords_t = sphere.coords(h)
    r = sphere.radius(coords_t).cpu().numpy()
    coords = coords_t.cpu().numpy()
    logits = model.from_hidden(h).cpu()
    hn = h.cpu().numpy()
    y = labels.numpy()

    out_aucs = [roc_auc_score(y[:, i], logits[:, i].numpy()) for i in range(8)]
    align_aucs = [roc_auc_score(y[:, f], coords[:, k])
                  for k, f in enumerate(ALIGNED_IDX)]
    country = y[:, COUNTRY_IDX]
    r1, r0 = r[country == 1].mean(), r[country == 0].mean()
    radius_auc = roc_auc_score(country, -r)
    lin_auc = quick_probe_auc(hn, country)

    singles, doubles, triples = [], [], []
    fa, fb, fc = ALIGNED_IDX
    for va in (0, 1):
        ma = y[:, fa] == va
        singles.append(quick_probe_auc(hn[ma], country[ma]))
        for vb in (0, 1):
            mb = y[:, fb] == vb
            singles.append(quick_probe_auc(hn[mb], country[mb]))
            doubles.append(quick_probe_auc(hn[ma & mb], country[ma & mb]))
            for vc in (0, 1):
                mc = y[:, fc] == vc
                singles.append(quick_probe_auc(hn[mc], country[mc]))
                doubles.append(quick_probe_auc(hn[ma & mc], country[ma & mc]))
                doubles.append(quick_probe_auc(hn[mb & mc], country[mb & mc]))
                m3 = ma & mb & mc
                triples.append(quick_probe_auc(hn[m3], country[m3]))
    max_single = np.nanmax(singles)
    max_double = np.nanmax(doubles)
    min_triple = np.nanmin(triples) if not np.all(np.isnan(triples)) else float("nan")

    print("  [out] " + " ".join(f"{n}={a:.3f}"
                                for n, a in zip(FEATURE_NAMES, out_aucs)))
    print(f"  [{tag}] out-min={min(out_aucs):.3f} "
          f"align={['%.3f' % a for a in align_aucs]} "
          f"r(c=1)={r1:.2f} r(c=0)={r0:.2f} -r AUC={radius_auc:.3f} "
          f"marg={lin_auc:.3f} max-1cond={max_single:.3f} "
          f"max-2cond={max_double:.3f} min-3cond={min_triple:.3f}")
    model.train()

    # criteria margins (positive = pass); score ranks checkpoints
    margins = [min(out_aucs) - 0.95, min(align_aucs) - 0.95, 0.65 - lin_auc,
               0.70 - max_single, 0.80 - max_double,
               (min_triple - 0.90) if not np.isnan(min_triple) else 0.0,
               radius_auc - 0.97]
    score = sum(m > 0 for m in margins) + sum(min(m, 0.02) for m in margins)
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
            scrub_scale = 1.0 - 0.9 * t          # 1.0 -> 0.1
        else:
            scrub_scale = 1.0

        order = torch.randperm(n, device=device)
        sums = dict(task=0.0, feat=0.0, geo=0.0, adv=0.0, fish=0.0, dm=0.0)
        n_batches, gate_open_batches = 0, 0
        for b in range(0, n, cfg["batch_size"]):
            idx = order[b: b + cfg["batch_size"]]
            x, lab = emb_tr_d[idx], lab_tr_d[idx]
            y = y_all[idx]

            h = model.hidden(x)

            # adversary probes train whenever the geometry stage has begun
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

            # model-side scrub losses only count once the probes are credible
            gate_open = (epoch >= cfg["warmup_epochs"]
                         and probe_ema > cfg["adv_gate_auc"])
            lam_adv = cfg["lambda_adv"] * scrub_scale if gate_open else 0.0
            lam_fish = cfg["lambda_fisher"] * scrub_scale if gate_open else 0.0

            # country logit reads h.detach(): the country head may DECODE what is
            # in h (the radius code) but must not SHAPE h, otherwise it keeps
            # writing a linear country code that no adversary can scrub.
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

            # distribution matching shapes the core geometry (it is feasible
            # jointly with the shells), so it follows the geometry ramp and is
            # neither probe-gated nor tapered.
            if lam_geo > 0:
                dm_loss = adv.dist_match_loss(coords, lab,
                                              frac=cfg["dist_frac"])
            else:
                dm_loss = h.sum() * 0
            lam_dm = geo_s * cfg["lambda_dm"]

            if lam_adv > 0:
                adv_loss = adv.confusion_loss(h, lab)
                fish_loss = adv.fisher_loss(h, lab)
            else:
                adv_loss = fish_loss = h.sum() * 0

            loss = (task + cfg["lambda_feat"] * feat + lam_geo * geo
                    + lam_adv * adv_loss + lam_fish * fish_loss
                    + lam_dm * dm_loss)
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                sphere.log_scale.clamp_(max=3.0)

            sums["task"] += task.item()
            sums["feat"] += feat.item()
            sums["geo"] += geo.item()
            sums["adv"] += adv_loss.detach().item()
            sums["fish"] += fish_loss.detach().item()
            sums["dm"] += dm_loss.detach().item()
            n_batches += 1
            gate_open_batches += int(gate_open)

        means = {k: v / n_batches for k, v in sums.items()}
        print(f"epoch {epoch:3d}  lam_geo={lam_geo:.2f} scrub={scrub_scale:.2f} "
              f"gate={gate_open_batches}/{n_batches} probe_ema={probe_ema:.3f}  "
              f"task={means['task']:.4f} feat={means['feat']:.4f} "
              f"geo={means['geo']:.4f} adv={means['adv']:.4f} "
              f"fish={means['fish']:.4f} dm={means['dm']:.4f}")
        if epoch % 5 == 4 or epoch == cfg["epochs"] - 1:
            score = epoch_report(model, sphere, emb_val, lab_val, device, "val")
            # only consider checkpoints once the geometry is fully ramped in
            if epoch >= cfg["warmup_epochs"] + cfg["ramp_epochs"] and score > best_score:
                best_score = score
                best_state = dict(
                    head={k: v.clone() for k, v in model.state_dict().items()},
                    sphere={k: v.clone() for k, v in sphere.state_dict().items()},
                    epoch=epoch)
                print(f"  ** new best checkpoint (score={score:.3f})")

    if best_state is None:   # fall back to final weights
        best_state = dict(head=model.state_dict(), sphere=sphere.state_dict(),
                          epoch=cfg["epochs"] - 1)
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
