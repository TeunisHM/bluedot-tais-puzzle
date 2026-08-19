# BlueDot TAIS Puzzle #1 — solution and a weirder model

My work on [BlueDot's Technical AI Safety Puzzle #1](https://github.com/SamDower/bluedot-tais-puzzle):
find the one feature that a small classifier does *not* represent linearly, explain the geometry it
uses instead, and then train a model with an even stranger representation.

**Short answer:** the non-linear feature is **`country`**, and the model stores it as a **radius** — an
annulus in the 2D plane spanned by the (linear) `food` and `sentiment` directions of layer 2.
My own model pushes that idea up a dimension, into a **spherical shell code** in 3D.

## The setup

The puzzle model is `sentence-transformers/all-MiniLM-L6-v2` (mean-pooled, 384-d) followed by a
5-layer MLP head with ReLUs and 8 sigmoid outputs, one per feature: `number`, `question`, `color`,
`food`, `sentiment`, `country`, `person`, `body_part`. The layer of interest is the **post-ReLU
activation of hidden layer 2**, `h = m.layers[:6](embeddings)`, in R^64.

![Model architecture](model_architecture.png)

## Task 1 — which feature is non-linear

A linear vs. MLP probe battery on `h` over the 1500 test examples. Seven features are cleanly linear;
one is not:

| feature | linear probe AUC | MLP probe AUC | gap |
|---|---|---|---|
| number | 0.997 | 0.997 | −0.000 |
| question | 1.000 | 1.000 | −0.000 |
| color | 0.999 | 0.998 | −0.000 |
| food | 0.996 | 0.991 | −0.005 |
| sentiment | 0.996 | 0.995 | −0.001 |
| **country** | **0.559** | **0.935** | **+0.376** |
| person | 1.000 | 0.998 | −0.002 |
| body_part | 0.998 | 0.999 | +0.000 |

`country` is fully decodable from `h` — just not by a hyperplane.

## Task 2 — how `country` is represented

Conditioning on a single other feature rescues the linear probe, and the rescuing features are
`food` and `sentiment` (not `number`, and not any XOR-style label combination — the labels are
essentially uncorrelated, `corr(food, sentiment) = −0.03`):

```
marginal linear AUC                 = 0.553
  conditioned on food=0             = 0.964   (+0.411)
  conditioned on food=1             = 0.963   (+0.410)
  conditioned on sentiment=0        = 0.904   (+0.351)
  conditioned on sentiment=1        = 0.927   (+0.374)
  conditioned on number=0           = 0.654   (+0.101)
```

The within-subgroup `country` directions point *opposite* ways across subgroups
(`cos(d_food=0, d_food=1) = −0.87`), which is the signature of a radial rather than linear code:
country=1 sits at small radius and country=0 spreads out in every direction, so each subgroup sees
the shell from its own side. Taking `x` and `y` to be the coordinates along the normalized `sentiment`
and `food` probe directions and forming the in-plane radius:

```python
r = np.sqrt(x**2 + y**2)
roc_auc_score(country, -r)   # 0.9933
```

**A single scalar — the distance from the origin in the (food, sentiment) plane — recovers `country`
at AUC 0.993, versus 0.559 for the best hyperplane.** Causal steering agrees: pushing `h` along the
`food` direction moves the `country` output in opposite directions depending on which side of the
plane the example started on, and clamping the coordinate to ±0.8 flips the `country` probability by
up to ~0.5 with the sign set by the example's `food` value.

The full argument, with the plots, is in [`BlueDot.ipynb`](BlueDot.ipynb).

## Task 3 — a weirder representation: a 3D spherical shell

If a 2D annulus is odd, a 3D shell is odder: I trained a head with the identical architecture in
which `country` lives **only** in the radius of a 3-dimensional subspace whose axes `u1, u2, u3`
linearly encode `food`, `sentiment` and `number`. `country=1` maps to a small shell (r ≈ 0.5),
`country=0` to a large one (r ≈ 2.0).

The point of the extra dimension: for the original 2D code, conditioning on *one* feature linearizes
`country`. For a 3D shell it does not — fixing two features still leaves the third coordinate ranging
over ±r, so country=0 mass stays on both sides of the small shell. You have to fix **all three**
(i.e. pick an octant) before a hyperplane works. Probe AUC therefore climbs monotonically with
conditioning depth, which is exactly the fingerprint of a radial code:

```
marginal 0.524  ->  single 0.849  ->  double 0.943  ->  octant 0.975
```

Results for the reported checkpoint (`sphere_model_v2b.pt`, test set):

| criterion | value | target |
|---|---|---|
| min output AUC across the 8 features | 0.9845 | ≥ 0.95 |
| min linear probe on `h`, 7 non-country features | 0.9923 | ≥ 0.95 |
| marginal linear probe for `country` | 0.5236 | ≤ 0.65 |
| mean octant-conditioned probe | 0.9753 | ≥ 0.90 |
| centered-radius ranking `-‖coords − c‖` | 0.9919 | ≥ 0.96 |

So the model is still a good 8-feature classifier, six of the other features stay linearly readable,
and `country` is invisible to any hyperplane while being almost perfectly readable from one radius.

## Repo layout

| file | what it is |
|---|---|
| `BlueDot.ipynb` | the tasks 1–2 analysis: probe battery, conditioned probes, the (food, sentiment) plane, radius test, causal steering |
| `analyze.py` | first pass at the puzzle model as a plain script |
| `sphere_model.py` | model definition + data/embedding utilities for the 3D spherical code |
| `train_sphere.py` | task-3 training run (2D-style criteria, `sphere_model.pt`) |
| `train_sphere_v2.py` | the training run that produced the reported model (`sphere_model_v2b.pt`) |
| `analyze_sphere.py`, `analyze_sphere_v2.py` | probe-ladder evaluation + diagnostic plots for a trained sphere model |
| `model.pt`, `feature_names.json`, `model_architecture.png`, `puzzle.ipynb` | unmodified files from the upstream puzzle repo |
| `sphere_model_v2b.pt` | the task-3 checkpoint the numbers above refer to |

Training logs, intermediate checkpoints, generated plots and probe-ladder dumps are regenerable and
are not tracked — see `.gitignore`.

## Reproducing

The puzzle's dataset is **not** included here. Fetch `data/train.jsonl` and `data/test.jsonl` from the
upstream repo and drop them in `data/`:

```bash
git clone https://github.com/SamDower/bluedot-tais-puzzle.git /tmp/puzzle
cp -r /tmp/puzzle/data .
```

Then:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# tasks 1-2
jupyter lab BlueDot.ipynb

# task 3: evaluate the shipped checkpoint (writes plots + a probe-ladder table)
python analyze_sphere_v2.py sphere_model_v2b.pt

# task 3: retrain from scratch (~150 epochs; first run embeds the dataset and caches it)
python train_sphere_v2.py
```

The first run downloads the MiniLM encoder from Hugging Face and caches sentence embeddings under
`cache/`. Training is CPU-friendly; the head is tiny and the encoder is only used once per split.

## Credits

The puzzle, `model.pt`, the dataset and the starter notebook are BlueDot Impact's, from
[SamDower/bluedot-tais-puzzle](https://github.com/SamDower/bluedot-tais-puzzle). Everything else here
is my own analysis and training code.
