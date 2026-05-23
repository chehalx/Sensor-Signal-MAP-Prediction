
import os, sys, pickle, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import load_data, TRAIN_PATH, TEST_PATH

MODELS_DIR = "../outputs/models"
PLOTS_DIR  = "../outputs/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# plot_2vectors exact from utils.py
def calc_mae(gt, pred):
    return np.mean(abs(np.array(gt) - np.array(pred)))

def plot_2vectors(label, pred, name):
    list1 = np.array(label)
    list2 = np.array(pred)
    mae   = calc_mae(list1, list2[:,0] if len(list2.shape)==2 else list2)
    sorted_id = sorted(range(len(list1)), key=lambda k: list1[k])
    plt.clf()
    plt.text(0, np.min(list2), f'MAE={mae:.4f}')
    plt.scatter(np.arange(list2.shape[0]), list2[sorted_id], s=1, alpha=0.5,
                label=f'{name} prediction', color='blue')
    plt.scatter(np.arange(list1.shape[0]), list1[sorted_id], s=1, alpha=0.5,
                label=f'{name} label', color='red')
    plt.legend()
    plt.title(f'MAP: Prediction vs Label — {name}  (MAE={mae:.4f})')
    path = os.path.join(PLOTS_DIR, f'{name}.png')
    plt.savefig(path); plt.clf()
    print(f'  Saved: {name}.png  (MAE={mae:.4f})')
    return mae

# Load data
print("Loading data...")
F_tr, F_te, y_tr, y_te, _ = load_data(TRAIN_PATH, TEST_PATH)
X_tr_raw = np.load(TRAIN_PATH)[:, :1000].astype(np.float32)
X_te_raw = np.load(TEST_PATH)[:,  :1000].astype(np.float32)

# Predict helpers 
def predict_sklearn(obj, F):
    if isinstance(obj, dict) and "model" in obj:
        p = obj["model"].predict(F).reshape(-1,1)
        return obj["qt"].inverse_transform(p).ravel()
    return obj.predict(F)

def predict_cnn(payload, X_raw, F_phys):
    import torch
    from model_cnn import MAPNet, normalise
    # use ref_std saved during training to align test amplitude to train
    ref_std = payload.get("ref_std", None)
    Xn  = normalise(X_raw, ref_std)
    m   = MAPNet(n_phys=payload["n_phys"])
    m.load_state_dict(payload["model_state"]); m.eval()
    with torch.no_grad():
        p = m(torch.tensor(Xn[:,None,:]),
              torch.tensor(F_phys.astype(np.float32))).numpy().reshape(-1,1)
    return payload["qt"].inverse_transform(p).ravel()

# Load all models & plot
pkls = sorted(f for f in os.listdir(MODELS_DIR) if f.endswith(".pkl"))
if not pkls:
    sys.exit(f"No models found in {MODELS_DIR}/. Run run_all.py first.")

results = {}
for fname in pkls:
    name = fname.replace(".pkl","").replace("_"," ").title()
    obj  = pickle.load(open(os.path.join(MODELS_DIR, fname), "rb"))

    if fname == "cnn.pkl":
        p_tr = predict_cnn(obj, X_tr_raw, F_tr)
        p_te = predict_cnn(obj, X_te_raw, F_te)
    else:
        p_tr = predict_sklearn(obj, F_tr)
        p_te = predict_sklearn(obj, F_te)

    tr_mae = calc_mae(y_tr, p_tr)
    te_mae = plot_2vectors(y_te, p_te, f"MAP_{name.replace(' ','_')}")
    results[name] = {"train": tr_mae, "test": te_mae}

# MAE bar chart
names = list(results.keys())
x     = np.arange(len(names))
fig, ax = plt.subplots(figsize=(12, 5))
bars = ax.bar(x-0.18, [results[n]["test"]  for n in names], 0.32,
              label="Test MAE",  color="tomato",    alpha=0.85)
ax.bar(       x+0.18, [results[n]["train"] for n in names], 0.32,
              label="Train MAE", color="steelblue", alpha=0.85)
[ax.text(b.get_x()+b.get_width()/2, results[n]["test"]+0.1,
         f"{results[n]['test']:.2f}", ha="center", fontsize=8)
 for b, n in zip(bars, names)]
ax.set_xticks(x); ax.set_xticklabels(names, rotation=15, ha="right")
ax.set(ylabel="MAE (mmHg)", title="Train vs Test MAE — all models")
ax.legend(); fig.tight_layout()
fig.savefig(f"{PLOTS_DIR}/mae_comparison.png", dpi=130); plt.close()
print(f"  Saved: mae_comparison.png")

# Summary
print("\n" + "=" * 50)
print(f"  {'Model':<20} {'Train':>8} {'Test':>8}")
print("  " + "-" * 40)
for name, r in sorted(results.items(), key=lambda x: x[1]["test"]):
    print(f"  {name:<20} {r['train']:>8.3f} {r['test']:>8.3f}")
print("=" * 50)
