#%%imports
import torch, torch.nn as nn
from sentence_transformers import SentenceTransformer
import json
import matplotlib.pyplot as plt 

#%% --- 1. Define the MLP head  ---
class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(384, 64), nn.ReLU(),   # hidden 0
            nn.Linear(64, 64),  nn.ReLU(),   # hidden 1
            nn.Linear(64, 64),  nn.ReLU(),   # hidden 2  ← non-linear activation here (post-ReLU)
            nn.Linear(64, 64),  nn.ReLU(),   # hidden 3
            nn.Linear(64, 8),                # logits
        )
    def forward(self, x):
        return self.layers(x)
    
#%% --- 2. Load encoder (downloaded from HF) and head (local file) ---
enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
m = Head()
m.load_state_dict(torch.load("model.pt", map_location="cpu", weights_only=False))
m.eval()

#%% --- 3. Get predictions ---

X = []

with open("data/test.jsonl") as f:
    for line in f:
        X.append(json.loads(line)["text"])

with torch.no_grad():
    embeddings = torch.from_numpy(
        enc.encode(X, convert_to_numpy=True)   # (N, 384), mean-pooled
    )
    logits = m(embeddings)                          # (N, 8)
    probs  = torch.sigmoid(logits)                  # (N, 8) — independent per feature
    preds  = (probs > 0.5).int()  

#%% 4. Get activations at the right spot (post-ReLU of hidden 2) ---
with torch.no_grad():
    layer2_acts = m.layers[:6](embeddings)          # (N, 64)

#%% 5.explore linearity
from sklearn.linear_model import LogisticRegression

for i in range(probs.shape[1]):
    clf = LogisticRegression(max_iter=1000)
    clf.fit(layer2_acts, preds[:,i])
    score = clf.score(layer2_acts, preds[:,i])
    print(f"Accuracy for {i}:, {score}")
    proj = layer2_acts @ clf.coef_[0]
    plt.scatter(proj, probs[:, i], s=5)
# %%