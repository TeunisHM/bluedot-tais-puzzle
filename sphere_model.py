"""Model definition + shared data utilities for the 3D spherical radial code experiment.

Goal: same Head architecture as the original puzzle model, but at hidden layer 2
(post-ReLU, h in R^64) the "country" feature is encoded ONLY in the radius
r = ||(h.u1, h.u2, h.u3)|| of a 3D subspace whose axes u1, u2, u3 linearly
encode food, sentiment and number respectively.

country=1 -> r near R_SMALL, country=0 -> r near R_LARGE.
"""
import json
import os

import torch
import torch.nn as nn

FEATURE_NAMES = ["number", "question", "color", "food", "sentiment", "country",
                 "person", "body_part"]
# u1 -> food, u2 -> sentiment, u3 -> number
ALIGNED_FEATURES = ["food", "sentiment", "number"]
ALIGNED_IDX = [FEATURE_NAMES.index(f) for f in ALIGNED_FEATURES]
COUNTRY_IDX = FEATURE_NAMES.index("country")
R_SMALL, R_LARGE = 0.5, 2.0  # country=1 -> small shell, country=0 -> large shell

HIDDEN_LAYER_SLICE = slice(0, 6)  # layers[:6] -> post-ReLU of hidden 2


class Head(nn.Module):
    """5-layer MLP head, identical to the original puzzle model."""

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(384, 64), nn.ReLU(),   # hidden 0
            nn.Linear(64, 64), nn.ReLU(),    # hidden 1
            nn.Linear(64, 64), nn.ReLU(),    # hidden 2  <- sculpted layer (post-ReLU)
            nn.Linear(64, 64), nn.ReLU(),    # hidden 3
            nn.Linear(64, 8),                # logits
        )

    def hidden(self, x):
        """Post-ReLU activations of hidden layer 2."""
        return self.layers[HIDDEN_LAYER_SLICE](x)

    def from_hidden(self, h):
        """Remaining layers: hidden-2 activations -> 8 logits."""
        return self.layers[6:](h)

    def forward(self, x):
        return self.layers(x)


class SphereCode(nn.Module):
    """Three learnable directions in activation space, re-orthonormalized by QR
    on every forward pass.

    coords(h) = h @ Q, with Q in R^{64x3} (columns u1, u2, u3).

    The alignment logits carry a learnable positive scale and bias per direction:
    on the small shell (r = 0.5) raw coordinates are at most 0.5, deep inside
    sigmoid's linear regime, so a plain BCE on sigmoid(h.u) would push coordinate
    magnitudes up and fight the radius loss. The scale lets BCE saturate at small
    magnitudes; linear-probe AUC on h.u is unaffected (scale/bias invariant).
    """

    def __init__(self, dim=64, k=3, init_scale=4.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(dim, k) / dim ** 0.5)
        self.log_scale = nn.Parameter(
            torch.full((k,), torch.tensor(init_scale).log().item()))
        self.bias = nn.Parameter(torch.zeros(k))
        # learnable shell center: h is post-ReLU (non-negative orthant), so the
        # activation cloud's natural center is offset from the origin; forcing
        # shells around the origin fights the geometry.
        self.center = nn.Parameter(torch.zeros(k))

    def directions(self):
        Q, R = torch.linalg.qr(self.W)
        # fix QR's column-sign ambiguity so directions don't flip between steps
        return Q * torch.sign(torch.diagonal(R)).unsqueeze(0)

    def coords(self, h):
        return h @ self.directions()

    def radius(self, coords):
        """Distance from the learned shell center."""
        return (coords - self.center).norm(dim=-1)

    def alignment_logits(self, coords):
        return coords * self.log_scale.exp() + self.bias


class AdversaryBank(nn.Module):
    """Linear probes that try to read country off h.

    probes[0] is marginal (all examples); probes[1..6] are conditional, one per
    (aligned feature, value 0/1) subset. They are trained on detached h; the
    model is trained to push their predictions back to chance. This both removes
    linear country leakage outside the sphere subspace and forces within-octant
    angular spread on the shells (otherwise conditioning on a single feature
    rescues a linear probe, as it does for the 2D radial code).
    """

    def __init__(self, dim=64):
        super().__init__()
        # subset i>0: feature ALIGNED_IDX[(i-1)//2], value (i-1)%2
        self.probes = nn.ModuleList([nn.Linear(dim, 1) for _ in range(7)])
        # running statistics so probes (and scrub losses) operate on
        # standardized activations, like the sklearn evaluation probes do.
        # Country gaps hiding in low-variance directions are invisible in raw
        # units but glaring after standardization.
        self.register_buffer("mu", torch.zeros(dim))
        self.register_buffer("sig", torch.ones(dim))

    @torch.no_grad()
    def update_stats(self, h, momentum=0.95):
        self.mu.mul_(momentum).add_(h.mean(0), alpha=1 - momentum)
        self.sig.mul_(momentum).add_(h.std(0) + 1e-3, alpha=1 - momentum)

    def standardize(self, h):
        return (h - self.mu) / self.sig

    @staticmethod
    def subset_mask(i, labels):
        """Boolean mask of examples probe i is responsible for."""
        if i == 0:
            return torch.ones(labels.shape[0], dtype=torch.bool,
                              device=labels.device)
        feat = ALIGNED_IDX[(i - 1) // 2]
        val = (i - 1) % 2
        return labels[:, feat] == val

    def probe_loss(self, h_detached, labels):
        """BCE of each probe predicting country on its subset (trains probes)."""
        y = labels[:, COUNTRY_IDX].float()
        hs = self.standardize(h_detached)
        total, n = 0.0, 0
        for i, probe in enumerate(self.probes):
            m = self.subset_mask(i, labels)
            if m.sum() < 8 or y[m].min() == y[m].max():
                continue
            logit = probe(hs[m]).squeeze(-1)
            total = total + nn.functional.binary_cross_entropy_with_logits(logit, y[m])
            n += 1
        return total / max(n, 1)

    def fisher_loss(self, x, labels, shrink=0.1, subsets=range(7)):
        """Fisher discriminant d^T (Sigma + eps I)^-1 d of country, marginal and
        within each single-feature subset, computed in closed form per batch.

        This is the statistic a (standardized) logistic probe actually exploits,
        so unlike fixed-metric moment matching it cannot be gamed by rescaling:
        raw-unit mean matching goes blind when the model inflates variance
        (observed: complement std grew ~7x while probes stayed at 0.97), and
        whitened mean matching over-penalizes low-variance directions. Here,
        inflating variance along the gap direction reduces the loss only by
        genuinely drowning the linear signal — which is the within-shell spread
        we want. No adversary lag either: the optimal linear probe is recomputed
        analytically every batch.
        """
        y = labels[:, COUNTRY_IDX]
        dim = x.shape[1]
        eye = torch.eye(dim, device=x.device, dtype=x.dtype)
        total, n = 0.0, 0
        for i in subsets:
            m = self.subset_mask(i, labels)
            m1, m0 = m & (y == 1), m & (y == 0)
            n1, n0 = int(m1.sum()), int(m0.sum())
            if n1 < 16 or n0 < 16:
                continue
            x1, x0 = x[m1], x[m0]
            d = x1.mean(0) - x0.mean(0)
            xc = torch.cat([x1 - x1.mean(0), x0 - x0.mean(0)])
            cov = xc.T @ xc / (len(xc) - 2)
            lam = shrink * cov.diagonal().mean()
            sol = torch.linalg.solve(cov + lam * eye, d)
            total = total + d @ sol
            n += 1
        return total / max(n, 1)

    def dist_match_loss(self, coords, labels, frac=0.6, n_q=15):
        """Capped conditional quantile matching on the sphere coordinates.

        For each aligned feature f_k = v (sign-correct the coordinate so "small"
        means near the decision boundary), force the bottom `frac` of the
        LARGE shell's conditional coordinate distribution to match the small
        shell's full distribution along that axis.

        Exact conditional distribution matching is geometrically impossible
        (all three coordinates small on every point would force r small), so
        the top 1-frac of the large shell stays free to carry the radius.
        A conditional linear probe along the conditioned axis then sees ~frac
        of large-shell points inside the small-shell range: best AUC is about
        1 - frac/2. The octant (triple-conditioned) rescue survives any such
        rearrangement because, within an octant, the sign-corrected coordinate
        sum is the l1 norm and ||c||_1 >= ||c||_2 = r: the all-ones diagonal
        probe always separates the shells.
        """
        y = labels[:, COUNTRY_IDX]
        grid = torch.linspace(0.05, 0.95, n_q, device=coords.device)
        total, n = 0.0, 0
        for k, fidx in enumerate(ALIGNED_IDX):
            for v in (0, 1):
                m = labels[:, fidx] == v
                m1, m0 = m & (y == 1), m & (y == 0)
                if m1.sum() < 12 or m0.sum() < 12:
                    continue
                sign = 1.0 if v == 1 else -1.0
                c_small = sign * coords[m1, k]   # small shell, full distribution
                c_large = sign * coords[m0, k]   # large shell, bottom frac only
                q_small = torch.quantile(c_small, grid)
                q_large = torch.quantile(c_large, grid * frac)
                total = total + ((q_large - q_small) ** 2).mean()
                n += 1
        return total / max(n, 1)

    def confusion_loss(self, h, labels, subsets=range(7)):
        """Push probe outputs toward 0.5 on their subsets (trains the model).

        Probe weights are detached so only the model receives gradient.
        """
        y = labels[:, COUNTRY_IDX].float()
        hs = self.standardize(h)   # buffers carry no grad; h keeps its grad
        total, n = 0.0, 0
        for i in subsets:
            probe = self.probes[i]
            m = self.subset_mask(i, labels)
            if m.sum() < 8 or y[m].min() == y[m].max():
                continue
            logit = hs[m] @ probe.weight.detach().T + probe.bias.detach()
            target = torch.full_like(logit, 0.5)
            total = total + nn.functional.binary_cross_entropy_with_logits(logit, target)
            n += 1
        return total / max(n, 1)


# --------------------------------------------------------------------------
# data utilities
# --------------------------------------------------------------------------

def load_jsonl(path):
    texts, labels = [], []
    with open(path) as f:
        for line in f:
            ex = json.loads(line)
            texts.append(ex["text"])
            labels.append(ex["labels"])
    return texts, torch.tensor(labels)


def cached_embeddings(split, device="cpu", root="."):
    """MiniLM mean-pooled embeddings for data/{split}.jsonl, cached on disk."""
    cache_path = os.path.join(root, "cache", f"emb_{split}.pt")
    texts, labels = load_jsonl(os.path.join(root, "data", f"{split}.jsonl"))
    if os.path.exists(cache_path):
        emb = torch.load(cache_path, map_location="cpu", weights_only=True)
    else:
        from sentence_transformers import SentenceTransformer
        enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2",
                                  device=device)
        emb = torch.from_numpy(
            enc.encode(texts, convert_to_numpy=True, batch_size=128,
                       show_progress_bar=True))
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(emb, cache_path)
    return emb, labels
