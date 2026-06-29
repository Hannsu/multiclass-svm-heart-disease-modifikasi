import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing         import MinMaxScaler
from sklearn.svm                   import SVC
from sklearn.multiclass            import OneVsRestClassifier, OneVsOneClassifier, OutputCodeClassifier
from sklearn.neural_network        import MLPClassifier
from sklearn.naive_bayes           import GaussianNB
from sklearn.ensemble              import AdaBoostClassifier
from sklearn.tree                  import DecisionTreeClassifier
from sklearn.metrics               import (confusion_matrix, precision_score,
                                           recall_score, f1_score, accuracy_score,
                                           classification_report)
from sklearn.model_selection       import train_test_split
from itertools                     import combinations

# 1. LOAD & PREPROCESS DATA
COLUMNS = ["age","sex","cp","trestbps","chol","fbs","restecg",
           "thalach","exang","oldpeak","slope","ca","thal","target"]

LABEL_MAP = {0: "Healthy", 1: "Sick-Low", 2: "Sick-Medium",
             3: "Sick-High", 4: "Sick-Serious"}

print("=" * 60)
print("  MULTICLASS SVM – KLASIFIKASI PENYAKIT JANTUNG KORONER")
print("  Uji Coba Modifikasi  |  Split: 80% Train / 20% Test")
print("=" * 60)

# Load data
df = pd.read_csv("processed.csv",
                 header=None, names=COLUMNS,
                 na_values=["?"])

# Drop rows with missing values
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)
print(f"\n[INFO] Data setelah pembersihan: {len(df)} baris")

X = df.drop("target", axis=1).values.astype(float)
y = df["target"].values.astype(int)

print("\n[INFO] Distribusi kelas:")
for k, v in sorted(pd.Series(y).value_counts().items()):
    print(f"  {LABEL_MAP[k]:14s}  →  {v} sampel")

# 2. NORMALISASI MIN-MAX
scaler = MinMaxScaler()
X_norm = scaler.fit_transform(X)
print("\n[INFO] Normalisasi Min-Max selesai")

# 3. PEMBAGIAN DATA  —  MODIFIKASI: 80% train / 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X_norm, y, test_size=0.20, random_state=42, stratify=y)

print(f"\n[INFO] Pembagian data (MODIFIKASI: 80/20):")
print(f"  Training : {len(X_train)} sampel")
print(f"  Testing  : {len(X_test)} sampel")

print("\n[INFO] Distribusi per kelas (Train / Test):")
for k in sorted(LABEL_MAP):
    tr = np.sum(y_train == k)
    te = np.sum(y_test  == k)
    print(f"  {LABEL_MAP[k]:14s}  →  Train={tr:3d}  |  Test={te:3d}")

# 4. BINARY TREE SVM  (BT-SVM)
#    Struktur pohon biner: kelas dipisah secara hierarkis
#    Level 0: {Healthy} vs {Sick-*}
#    Level 1: {Sick-Low} vs {Sick-Medium, Sick-High, Sick-Serious}
#    Level 2: {Sick-Medium} vs {Sick-High, Sick-Serious}
#    Level 3: {Sick-High} vs {Sick-Serious}
class BTSVMClassifier:
    """Binary Tree SVM — hierarki 4 binary SVM."""
    def __init__(self, **svc_kwargs):
        self.svc_kwargs = svc_kwargs
        self.models = {}

    def _train_node(self, X, y, pos_classes, neg_classes):
        y_bin = np.where(np.isin(y, pos_classes), 1, -1)
        m = SVC(**self.svc_kwargs, probability=True)
        m.fit(X, y_bin)
        return m

    def fit(self, X, y):
        # Node 0: Healthy(0) vs Sick(1,2,3,4)
        self.models["n0"] = self._train_node(X, y, [0], [1,2,3,4])
        # Node 1: Sick-Low(1) vs Sick-Med/High/Serious(2,3,4)
        mask1 = np.isin(y, [1,2,3,4])
        self.models["n1"] = self._train_node(X[mask1], y[mask1], [1], [2,3,4])
        # Node 2: Sick-Med(2) vs Sick-High/Serious(3,4)
        mask2 = np.isin(y, [2,3,4])
        self.models["n2"] = self._train_node(X[mask2], y[mask2], [2], [3,4])
        # Node 3: Sick-High(3) vs Sick-Serious(4)
        mask3 = np.isin(y, [3,4])
        self.models["n3"] = self._train_node(X[mask3], y[mask3], [3], [4])
        return self

    def predict(self, X):
        preds = []
        for xi in X:
            xi = xi.reshape(1, -1)
            if self.models["n0"].predict(xi)[0] == 1:   # Healthy
                preds.append(0)
            else:                                         # Sick-*
                if self.models["n1"].predict(xi)[0] == 1:
                    preds.append(1)                       # Sick-Low
                else:
                    if self.models["n2"].predict(xi)[0] == 1:
                        preds.append(2)                   # Sick-Medium
                    else:
                        if self.models["n3"].predict(xi)[0] == 1:
                            preds.append(3)               # Sick-High
                        else:
                            preds.append(4)               # Sick-Serious
        return np.array(preds)

# 5. DDAG-SVM  (Directed Acyclic Graph)
#    Menggunakan n*(n-1)/2 binary SVM seperti OAO,
#    tetapi keputusan diambil lewat path graf DAG.
class DDAGSVMClassifier:
    """DDAG-SVM: binary SVM untuk setiap pasang kelas,
       keputusan via rooted binary DAG."""
    def __init__(self, **svc_kwargs):
        self.svc_kwargs = svc_kwargs
        self.classifiers = {}
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        for (ci, cj) in combinations(self.classes_, 2):
            mask = np.isin(y, [ci, cj])
            Xs, ys = X[mask], y[mask]
            m = SVC(**self.svc_kwargs)
            m.fit(Xs, ys)
            self.classifiers[(ci, cj)] = m
        return self

    def _predict_one(self, x):
        candidates = list(self.classes_)
        while len(candidates) > 1:
            ci, cj = candidates[0], candidates[-1]
            key = (ci, cj) if (ci, cj) in self.classifiers else (cj, ci)
            pred = self.classifiers[key].predict(x.reshape(1, -1))[0]
            if pred == ci:
                candidates.pop()
            else:
                candidates.pop(0)
        return candidates[0]

    def predict(self, X):
        return np.array([self._predict_one(x) for x in X])

# 6. DEFINISI SEMUA MODEL
SVC_PARAMS = dict(kernel="rbf", C=1.0, gamma="scale", random_state=42)

models = {
    "BT-SVM" : BTSVMClassifier(**SVC_PARAMS),
    "OAA-SVM": OneVsRestClassifier(SVC(**SVC_PARAMS)),
    "OAO-SVM": OneVsOneClassifier(SVC(**SVC_PARAMS)),
    "DDAG-SVM": DDAGSVMClassifier(**SVC_PARAMS),
    "ECOC-SVM": OutputCodeClassifier(SVC(**SVC_PARAMS), code_size=2, random_state=42),
    "MLP"    : MLPClassifier(hidden_layer_sizes=(100,), max_iter=500,
                             random_state=42),
    "NB"     : GaussianNB(),
    "ESB"    : AdaBoostClassifier(
                   estimator=DecisionTreeClassifier(max_depth=1),
                   n_estimators=100, random_state=42),
}

CLASSES    = [0, 1, 2, 3, 4]
CLASS_NAMES = [LABEL_MAP[c] for c in CLASSES]

# 7. TRAINING & EVALUASI
results = {}   # {method: {class: {recall, precision, f1}}, overall_acc}

print("\n" + "=" * 60)
print("  TRAINING & EVALUASI MODEL")
print("=" * 60)

for name, model in models.items():
    print(f"\n[{name}] Training ...", end=" ")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("selesai.")

    # Per-class metrics
    r  = recall_score   (y_test, y_pred, labels=CLASSES, average=None,
                         zero_division=0)
    p  = precision_score(y_test, y_pred, labels=CLASSES, average=None,
                         zero_division=0)
    f  = f1_score       (y_test, y_pred, labels=CLASSES, average=None,
                         zero_division=0)
    oa = accuracy_score (y_test, y_pred)

    results[name] = {
        "recall"  : r * 100,
        "precision": p * 100,
        "f1"      : f * 100,
        "overall_acc": oa * 100,
    }

    print(f"  Overall Accuracy: {oa*100:.3f}%")
    for i, c in enumerate(CLASS_NAMES):
        print(f"  {c:14s} → Recall={r[i]*100:6.2f}%  "
              f"Precision={p[i]*100:6.2f}%  F1={f[i]*100:6.2f}%")

# 8. SUMMARY TABLE
print("\n" + "=" * 60)
print("  RINGKASAN HASIL UJI COBA MODIFIKASI  (80/20 split)")
print("=" * 60)

# Overall accuracy table
print("\n[Overall Accuracy]")
print(f"  {'Method':<12} {'Accuracy':>10}")
print("  " + "-" * 24)
for name in models:
    print(f"  {name:<12} {results[name]['overall_acc']:>9.3f}%")

# F-Measure table
print("\n[F-Measure per Class (%)]")
header = f"  {'Method':<12}" + "".join(f"{c:>14}" for c in CLASS_NAMES)
print(header)
print("  " + "-" * (12 + 14 * len(CLASS_NAMES)))
for name in models:
    row = f"  {name:<12}"
    for i in range(len(CLASSES)):
        row += f"{results[name]['f1'][i]:>13.3f}%"
    print(row)

# 9. PERBANDINGAN DENGAN ARTIKEL (70/30 split)
article_acc = {
    "BT-SVM" : 61.86, "OAA-SVM": 56.70, "OAO-SVM": 51.546,
    "DDAG-SVM": 53.608, "ECOC-SVM": 58.763,
    "NB"     : 56.701, "ESB"    : 55.67,  "MLP"    : 58.763,
}
article_f1 = {
    "BT-SVM" : [86.793, 24.000, 33.333, 45.455, 18.182],
    "OAA-SVM": [86.793, 21.739, 32.000,  0.000,  0.000],
    "OAO-SVM": [80.851, 23.529, 21.622, 30.000, 22.222],
    "DDAG-SVM":[80.851, 25.000, 21.622, 45.455, 22.223],
    "ECOC-SVM":[76.200, 14.800,  0.000, 66.700,  0.000],
    "NB"     : [None]*5, "ESB": [None]*5, "MLP": [None]*5,
}

print("\n" + "=" * 60)
print("  PERBANDINGAN UJI COBA MODIFIKASI vs ARTIKEL")
print("  (Modifikasi: 80/20 | Artikel: 70/30)")
print("=" * 60)
print(f"\n  {'Method':<12} {'Artikel (%)':>13} {'Modifikasi (%)':>16} {'Δ':>8}")
print("  " + "-" * 52)
for name in models:
    art = article_acc.get(name, None)
    mod = results[name]["overall_acc"]
    delta = f"{mod - art:+.3f}" if art is not None else "—"
    art_str = f"{art:.3f}" if art is not None else "—"
    print(f"  {name:<12} {art_str:>13} {mod:>15.3f}% {delta:>8}")

# 10. VISUALISASI
METHOD_NAMES = list(models.keys())
COLORS = {"Recall": "#c0392b", "Precision": "#922b21", "F1-measure": "#5d6d7e"}
BG = "#fdf5f5"

def plot_class_performance(class_idx, class_name, ax, mod_results, method_names):
    """Bar chart: Recall / Precision / F1 per method untuk 1 kelas."""
    x   = np.arange(len(method_names))
    w   = 0.25
    rec = [mod_results[m]["recall"][class_idx]    for m in method_names]
    pre = [mod_results[m]["precision"][class_idx] for m in method_names]
    f1  = [mod_results[m]["f1"][class_idx]        for m in method_names]

    ax.bar(x - w, rec, w, label="Recall (%)",    color=COLORS["Recall"],    alpha=0.9)
    ax.bar(x,     pre, w, label="Precision (%)", color=COLORS["Precision"], alpha=0.9)
    ax.bar(x + w, f1,  w, label="F-measure (%)",color=COLORS["F1-measure"],alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(method_names, fontsize=8, rotation=15)
    ax.set_ylim(0, 110)
    ax.set_yticks(range(0, 101, 20))
    ax.set_title(class_name, fontweight="bold", fontsize=11, color="#8e1c1c", pad=6)
    ax.set_ylabel("(%)", fontsize=8)
    ax.set_facecolor(BG)
    ax.spines[["top","right"]].set_visible(False)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(axis="y", alpha=0.3, linestyle="--")

# Figure 1: Per-class performance (5 subplots + overall)
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.patch.set_facecolor(BG)
fig.suptitle("Hasil Uji Coba Modifikasi – Recall, Precision & F-Measure per Kelas\n"
             "(Pembagian Data: 80% Training / 20% Testing)",
             fontsize=14, fontweight="bold", color="#8e1c1c", y=1.01)

class_positions = [(0,0),(0,1),(0,2),(1,0),(1,1)]
for idx, (r, c) in enumerate(class_positions):
    plot_class_performance(idx, CLASS_NAMES[idx], axes[r][c],
                           results, METHOD_NAMES)

# Overall accuracy subplot
ax_oa = axes[1][2]
oa_vals = [results[m]["overall_acc"] for m in METHOD_NAMES]
bars = ax_oa.bar(METHOD_NAMES, oa_vals, color="#c0392b", alpha=0.85, edgecolor="#8e1c1c")
ax_oa.set_ylim(0, 80)
ax_oa.set_yticks(range(0, 81, 10))
ax_oa.set_title("Overall Accuracy", fontweight="bold", fontsize=11,
                color="#8e1c1c", pad=6)
ax_oa.set_ylabel("Accuracy (%)", fontsize=8)
ax_oa.set_facecolor(BG)
ax_oa.spines[["top","right"]].set_visible(False)
ax_oa.grid(axis="y", alpha=0.3, linestyle="--")
ax_oa.tick_params(axis="x", rotation=25, labelsize=8)
for bar, val in zip(bars, oa_vals):
    ax_oa.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
               f"{val:.1f}%", ha="center", va="bottom", fontsize=7,
               fontweight="bold", color="#8e1c1c")

plt.tight_layout()

plt.savefig("01_hasil_uji_coba_per_kelas.png",
            dpi=300,
            bbox_inches="tight")
plt.close()
print("\n[SAVED] 01_hasil_uji_coba_per_kelas.png")

# Figure 2: Perbandingan Overall Accuracy (Artikel vs Modifikasi)
fig2, ax2 = plt.subplots(figsize=(12, 6))
fig2.patch.set_facecolor(BG)
ax2.set_facecolor(BG)
x = np.arange(len(METHOD_NAMES))
w = 0.35
art_vals = [article_acc.get(m, 0) for m in METHOD_NAMES]
mod_vals = [results[m]["overall_acc"]  for m in METHOD_NAMES]

b1 = ax2.bar(x - w/2, art_vals, w, label="Artikel (70/30)",
             color="#c0392b", alpha=0.85, edgecolor="#8e1c1c")
b2 = ax2.bar(x + w/2, mod_vals, w, label="Modifikasi (80/20)",
             color="#5d6d7e", alpha=0.85, edgecolor="#2c3e50")

ax2.set_xticks(x)
ax2.set_xticklabels(METHOD_NAMES, fontsize=10)
ax2.set_ylim(0, 80)
ax2.set_yticks(range(0, 81, 10))
ax2.set_ylabel("Overall Accuracy (%)", fontsize=11)
ax2.set_title("Perbandingan Overall Accuracy\nArtikel (70/30) vs Uji Coba Modifikasi (80/20)",
              fontweight="bold", fontsize=13, color="#8e1c1c")
ax2.legend(fontsize=10)
ax2.spines[["top","right"]].set_visible(False)
ax2.grid(axis="y", alpha=0.3, linestyle="--")

for bar, val in zip(list(b1)+list(b2), art_vals+mod_vals):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f"{val:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

plt.tight_layout()
plt.savefig("02_perbandingan_akurasi_artikel_vs_modifikasi.png",
            dpi=300,
            bbox_inches="tight")
plt.close()
print("[SAVED] 02_perbandingan_akurasi_artikel_vs_modifikasi.png")

# Figure 3: F-Measure comparison (Artikel BT-SVM vs Modifikasi BT-SVM)
fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))
fig3.patch.set_facecolor(BG)
fig3.suptitle("Perbandingan F-Measure per Kelas – BT-SVM\n"
              "Artikel (70/30) vs Modifikasi (80/20)",
              fontweight="bold", color="#8e1c1c", fontsize=13)

art_f1_bt = [86.793, 24.000, 33.333, 45.455, 18.182]
mod_f1_bt = list(results["BT-SVM"]["f1"])
short_names = ["Healthy","Sick\nLow","Sick\nMed","Sick\nHigh","Sick\nSer"]
x3 = np.arange(len(short_names))
w3 = 0.35

for ax3, vals, title, clr in [
        (axes3[0], art_f1_bt, "Artikel  (70/30)", "#c0392b"),
        (axes3[1], mod_f1_bt, "Modifikasi  (80/20)", "#5d6d7e")]:
    bars3 = ax3.bar(x3, vals, color=clr, alpha=0.85, edgecolor="white", width=0.55)
    ax3.set_xticks(x3)
    ax3.set_xticklabels(short_names, fontsize=10)
    ax3.set_ylim(0, 110)
    ax3.set_yticks(range(0, 101, 20))
    ax3.set_ylabel("F-Measure (%)", fontsize=10)
    ax3.set_title(title, fontweight="bold", fontsize=12)
    ax3.set_facecolor(BG)
    ax3.spines[["top","right"]].set_visible(False)
    ax3.grid(axis="y", alpha=0.3, linestyle="--")
    for bar, val in zip(bars3, vals):
        ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=9,
                 fontweight="bold", color=clr)

plt.tight_layout()
plt.savefig("03_confusion_matrix.png",
            dpi=300,
            bbox_inches="tight")
plt.close()
print("[SAVED] 03_fmeasure_btsvm_artikel_vs_modifikasi.png")

# Figure 4: Distribusi data (Train vs Test per kelas – modifikasi)
fig4, ax4 = plt.subplots(figsize=(10, 5))
fig4.patch.set_facecolor(BG)
ax4.set_facecolor(BG)
tr_counts = [np.sum(y_train == k) for k in CLASSES]
te_counts = [np.sum(y_test  == k) for k in CLASSES]
x4 = np.arange(len(CLASS_NAMES))
ax4.bar(x4 - 0.2, tr_counts, 0.35, label="Training (80%)",
        color="#c0392b", alpha=0.85, edgecolor="#8e1c1c")
ax4.bar(x4 + 0.2, te_counts, 0.35, label="Testing (20%)",
        color="#5d6d7e", alpha=0.85, edgecolor="#2c3e50")
ax4.set_xticks(x4)
ax4.set_xticklabels(CLASS_NAMES, fontsize=10)
ax4.set_ylabel("Jumlah Sampel", fontsize=11)
ax4.set_title("Distribusi Dataset per Kelas – Uji Coba Modifikasi (80/20)",
              fontweight="bold", fontsize=12, color="#8e1c1c")
ax4.legend(fontsize=10)
ax4.spines[["top","right"]].set_visible(False)
ax4.grid(axis="y", alpha=0.3, linestyle="--")
for i, (tr, te) in enumerate(zip(tr_counts, te_counts)):
    ax4.text(i-0.2, tr+0.3, str(tr), ha="center", va="bottom",
             fontsize=9, fontweight="bold", color="#c0392b")
    ax4.text(i+0.2, te+0.3, str(te), ha="center", va="bottom",
             fontsize=9, fontweight="bold", color="#5d6d7e")
plt.tight_layout()
plt.savefig("04_grafik_akurasi.png",
            dpi=300,
            bbox_inches="tight")
plt.close()
print("[SAVED] 04_distribusi_dataset_modifikasi.png")

print("\n" + "=" * 60)
print("  SEMUA ANALISIS SELESAI — 4 GRAFIK TERSIMPAN")
print("=" * 60)