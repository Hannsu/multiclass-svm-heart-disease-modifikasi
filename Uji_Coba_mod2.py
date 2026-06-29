# Quick fix: run cv scores manually without cross_val_score for custom classes

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing    import MinMaxScaler
from sklearn.svm              import SVC
from sklearn.multiclass       import OneVsRestClassifier, OneVsOneClassifier, OutputCodeClassifier
from sklearn.neural_network   import MLPClassifier
from sklearn.naive_bayes      import GaussianNB
from sklearn.ensemble         import AdaBoostClassifier
from sklearn.tree             import DecisionTreeClassifier
from sklearn.metrics          import (confusion_matrix, precision_score,
                                      recall_score, f1_score, accuracy_score)
from sklearn.model_selection  import train_test_split, StratifiedKFold
from itertools                import combinations

BG       = "#fdf5f5"
RED      = "#c0392b"
RED_DARK = "#8e1c1c"
GREY     = "#5d6d7e"
COLORS   = {"Recall": RED, "Precision": "#922b21", "F1-measure": GREY}

def manual_cv(model_factory, X, y, cv=5):
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = []
    for train_idx, val_idx in skf.split(X, y):
        m = model_factory()
        m.fit(X[train_idx], y[train_idx])
        scores.append(accuracy_score(y[val_idx], m.predict(X[val_idx])))
    return np.array(scores)

print("=" * 60)
print("  MULTICLASS SVM – UJI COBA MODIFIKASI 2 (KAGGLE)")
print("  Dataset: heart.csv  |  Split: 80% Train / 20% Test")
print("=" * 60)

df = pd.read_csv("heart.csv")
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)
print(f"\n[INFO] Dataset Kaggle: {len(df)} baris")

LABEL_MAP   = {0: "Healthy", 1: "Sick"}
CLASS_NAMES = ["Healthy", "Sick"]
CLASSES     = [0, 1]

X = df.drop("target", axis=1).values.astype(float)
y = df["target"].values.astype(int)

scaler = MinMaxScaler()
X_norm = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_norm, y, test_size=0.20, random_state=42, stratify=y)

print(f"Training: {len(X_train)} | Testing: {len(X_test)}")
for k in CLASSES:
    print(f"  {LABEL_MAP[k]}: Train={np.sum(y_train==k)} | Test={np.sum(y_test==k)}")

class BTSVMClassifier:
    def __init__(self):
        self.model = None
    def fit(self, X, y):
        y_bin = np.where(y == 0, 1, -1)
        self.model = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42, probability=True)
        self.model.fit(X, y_bin)
        return self
    def predict(self, X):
        return np.where(self.model.predict(X) == 1, 0, 1)

class DDAGSVMClassifier:
    def __init__(self):
        self.model = None
    def fit(self, X, y):
        self.model = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)
        self.model.fit(X, y)
        return self
    def predict(self, X):
        return self.model.predict(X)

SVC_P = dict(kernel="rbf", C=1.0, gamma="scale", random_state=42)

def make_models():
    return {
        "BT-SVM"  : BTSVMClassifier(),
        "OAA-SVM" : OneVsRestClassifier(SVC(**SVC_P)),
        "OAO-SVM" : OneVsOneClassifier(SVC(**SVC_P)),
        "DDAG-SVM": DDAGSVMClassifier(),
        "ECOC-SVM": OutputCodeClassifier(SVC(**SVC_P), code_size=2, random_state=42),
        "MLP"     : MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42),
        "NB"      : GaussianNB(),
        "ESB"     : AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1),
                                       n_estimators=100, random_state=42),
    }

factories = {
    "BT-SVM"  : BTSVMClassifier,
    "OAA-SVM" : lambda: OneVsRestClassifier(SVC(**SVC_P)),
    "OAO-SVM" : lambda: OneVsOneClassifier(SVC(**SVC_P)),
    "DDAG-SVM": DDAGSVMClassifier,
    "ECOC-SVM": lambda: OutputCodeClassifier(SVC(**SVC_P), code_size=2, random_state=42),
    "MLP"     : lambda: MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42),
    "NB"      : GaussianNB,
    "ESB"     : lambda: AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1),
                                           n_estimators=100, random_state=42),
}

models = make_models()
results = {}

print("\n" + "=" * 60)
print("  TRAINING & EVALUASI")
print("=" * 60)

for name, model in models.items():
    print(f"\n[{name}] Training ...", end=" ")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("selesai.")

    r  = recall_score   (y_test, y_pred, labels=CLASSES, average=None, zero_division=0)
    p  = precision_score(y_test, y_pred, labels=CLASSES, average=None, zero_division=0)
    f  = f1_score       (y_test, y_pred, labels=CLASSES, average=None, zero_division=0)
    oa = accuracy_score (y_test, y_pred) * 100

    cv_sc = manual_cv(factories[name], X_norm, y, cv=5)

    results[name] = {
        "recall": r*100, "precision": p*100, "f1": f*100,
        "overall_acc": oa, "cv_mean": cv_sc.mean()*100, "cv_std": cv_sc.std()*100,
        "cm": confusion_matrix(y_test, y_pred, labels=CLASSES),
    }
    print(f"  Acc={oa:.1f}% | CV={cv_sc.mean()*100:.1f}%±{cv_sc.std()*100:.1f}%")

METHOD_NAMES = list(models.keys())

# ── GAMBAR 1: Per-kelas ──
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.patch.set_facecolor(BG)
fig.suptitle("Hasil Uji Coba Modifikasi 2 (Kaggle) – Recall, Precision & F-Measure per Kelas\n"
             "(Pembagian Data: 80% Training / 20% Testing)",
             fontsize=13, fontweight="bold", color=RED_DARK, y=1.01)

for ci, (ax, cn) in enumerate(zip(axes[:2], CLASS_NAMES)):
    x = np.arange(len(METHOD_NAMES)); w = 0.25
    ax.bar(x-w, [results[m]["recall"][ci]    for m in METHOD_NAMES], w, label="Recall (%)",    color=COLORS["Recall"],    alpha=0.9)
    ax.bar(x,   [results[m]["precision"][ci] for m in METHOD_NAMES], w, label="Precision (%)", color=COLORS["Precision"], alpha=0.9)
    ax.bar(x+w, [results[m]["f1"][ci]        for m in METHOD_NAMES], w, label="F-measure (%)", color=COLORS["F1-measure"],alpha=0.9)
    ax.set_xticks(x); ax.set_xticklabels(METHOD_NAMES, fontsize=8, rotation=20)
    ax.set_ylim(0, 115); ax.set_title(cn, fontweight="bold", color=RED_DARK, fontsize=11)
    ax.set_ylabel("(%)"); ax.set_facecolor(BG)
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linestyle="--"); ax.legend(fontsize=7)

ax_oa = axes[2]
oa_vals = [results[m]["overall_acc"] for m in METHOD_NAMES]
bars = ax_oa.bar(METHOD_NAMES, oa_vals, color=RED, alpha=0.85, edgecolor=RED_DARK)
ax_oa.set_ylim(0, 110); ax_oa.set_title("Overall Accuracy", fontweight="bold", color=RED_DARK, fontsize=11)
ax_oa.set_ylabel("Accuracy (%)"); ax_oa.set_facecolor(BG)
ax_oa.spines[["top","right"]].set_visible(False); ax_oa.grid(axis="y", alpha=0.3, linestyle="--")
ax_oa.tick_params(axis="x", rotation=25, labelsize=8)
for bar, val in zip(bars, oa_vals):
    ax_oa.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
               f"{val:.1f}%", ha="center", va="bottom", fontsize=7, fontweight="bold", color=RED_DARK)
plt.tight_layout()
plt.savefig("mod2_01_hasil_per_kelas.png", dpi=300, bbox_inches="tight")
plt.close(); print("\n[SAVED] mod2_01_hasil_per_kelas.png")

# ── GAMBAR 2: Perbandingan Akurasi ──
artikel_acc = {"BT-SVM":61.9,"OAA-SVM":56.7,"OAO-SVM":51.5,
               "DDAG-SVM":53.6,"ECOC-SVM":58.8,"MLP":58.8,"NB":56.7,"ESB":55.7}

fig2, ax2 = plt.subplots(figsize=(13, 6))
fig2.patch.set_facecolor(BG); ax2.set_facecolor(BG)
x = np.arange(len(METHOD_NAMES)); w = 0.35
art_vals  = [artikel_acc.get(m, 0)         for m in METHOD_NAMES]
mod2_vals = [results[m]["overall_acc"]      for m in METHOD_NAMES]
b1 = ax2.bar(x-w/2, art_vals,  w, label="Artikel (Cleveland, 80/20)", color=RED, alpha=0.85, edgecolor=RED_DARK)
b2 = ax2.bar(x+w/2, mod2_vals, w, label="Modifikasi 2 (Kaggle, 80/20)", color=GREY, alpha=0.85, edgecolor="#2c3e50")
ax2.set_xticks(x); ax2.set_xticklabels(METHOD_NAMES, fontsize=10)
ax2.set_ylim(0, 110); ax2.set_ylabel("Overall Accuracy (%)", fontsize=11)
ax2.set_title("Perbandingan Overall Accuracy\nArtikel (Cleveland) vs Uji Coba Modifikasi 2 (Kaggle)",
              fontweight="bold", fontsize=13, color=RED_DARK)
ax2.legend(fontsize=10); ax2.spines[["top","right"]].set_visible(False)
ax2.grid(axis="y", alpha=0.3, linestyle="--")
for bar, val in zip(list(b1)+list(b2), art_vals+mod2_vals):
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
             f"{val:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
plt.tight_layout()
plt.savefig("mod2_02_perbandingan_akurasi.png", dpi=300, bbox_inches="tight")
plt.close(); print("[SAVED] mod2_02_perbandingan_akurasi.png")

# ── GAMBAR 3: F-Measure per kelas ──
fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))
fig3.patch.set_facecolor(BG)
fig3.suptitle("F-Measure per Kelas – Uji Coba Modifikasi 2 (Kaggle)",
              fontweight="bold", color=RED_DARK, fontsize=13)
x3 = np.arange(len(METHOD_NAMES))
for ax3, ci, title, clr in [(axes3[0],0,"F-Measure – Kelas Healthy",RED),(axes3[1],1,"F-Measure – Kelas Sick",GREY)]:
    vals = [results[m]["f1"][ci] for m in METHOD_NAMES]
    bars3 = ax3.bar(x3, vals, color=clr, alpha=0.85, edgecolor="white", width=0.55)
    ax3.set_xticks(x3); ax3.set_xticklabels(METHOD_NAMES, fontsize=8, rotation=20)
    ax3.set_ylim(0, 115); ax3.set_ylabel("F-Measure (%)", fontsize=10)
    ax3.set_title(title, fontweight="bold", fontsize=11)
    ax3.set_facecolor(BG); ax3.spines[["top","right"]].set_visible(False)
    ax3.grid(axis="y", alpha=0.3, linestyle="--")
    for bar, val in zip(bars3, vals):
        ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold", color=clr)
plt.tight_layout()
plt.savefig("mod2_03_fmeasure_kelas.png", dpi=300, bbox_inches="tight")
plt.close(); print("[SAVED] mod2_03_fmeasure_kelas.png")

# ── GAMBAR 4: Distribusi Dataset ──
fig4, ax4 = plt.subplots(figsize=(8, 5))
fig4.patch.set_facecolor(BG); ax4.set_facecolor(BG)
tr_c = [np.sum(y_train==k) for k in CLASSES]; te_c = [np.sum(y_test==k) for k in CLASSES]
x4 = np.arange(2)
ax4.bar(x4-0.2, tr_c, 0.35, label="Training (80%)", color=RED, alpha=0.85, edgecolor=RED_DARK)
ax4.bar(x4+0.2, te_c, 0.35, label="Testing (20%)",  color=GREY, alpha=0.85, edgecolor="#2c3e50")
ax4.set_xticks(x4); ax4.set_xticklabels(CLASS_NAMES, fontsize=11)
ax4.set_ylabel("Jumlah Sampel", fontsize=11)
ax4.set_title("Distribusi Dataset Kaggle – Uji Coba Modifikasi 2 (80/20)",
              fontweight="bold", fontsize=12, color=RED_DARK)
ax4.legend(fontsize=10); ax4.spines[["top","right"]].set_visible(False)
ax4.grid(axis="y", alpha=0.3, linestyle="--")
for i,(tr,te) in enumerate(zip(tr_c, te_c)):
    ax4.text(i-0.2, tr+1, str(tr), ha="center", va="bottom", fontsize=10, fontweight="bold", color=RED)
    ax4.text(i+0.2, te+1, str(te), ha="center", va="bottom", fontsize=10, fontweight="bold", color=GREY)
plt.tight_layout()
plt.savefig("mod2_04_distribusi_dataset.png", dpi=300, bbox_inches="tight")
plt.close(); print("[SAVED] mod2_04_distribusi_dataset.png")

# ── GAMBAR 5: Stabilitas CV ──
fig5, ax5 = plt.subplots(figsize=(12, 5))
fig5.patch.set_facecolor(BG); ax5.set_facecolor(BG)
cv_m = [results[m]["cv_mean"] for m in METHOD_NAMES]
cv_s = [results[m]["cv_std"]  for m in METHOD_NAMES]
x5 = np.arange(len(METHOD_NAMES))
bars5 = ax5.bar(x5, cv_m, 0.5, yerr=cv_s, capsize=6, color=RED, alpha=0.85, edgecolor=RED_DARK,
                error_kw={"elinewidth":2,"ecolor":GREY})
ax5.set_xticks(x5); ax5.set_xticklabels(METHOD_NAMES, fontsize=10)
ax5.set_ylim(0, 110); ax5.set_ylabel("Accuracy (%)", fontsize=11)
ax5.set_title("Analisis Stabilitas Metode – 5-Fold Cross Validation (Kaggle)\nBar = Mean | Error Bar = Std Dev",
              fontweight="bold", fontsize=12, color=RED_DARK)
ax5.spines[["top","right"]].set_visible(False); ax5.grid(axis="y", alpha=0.3, linestyle="--")
for bar, m, s in zip(bars5, cv_m, cv_s):
    ax5.text(bar.get_x()+bar.get_width()/2, bar.get_height()+s+1,
             f"{m:.1f}%\n±{s:.1f}%", ha="center", va="bottom", fontsize=7.5, fontweight="bold", color=RED_DARK)
plt.tight_layout()
plt.savefig("mod2_05_stabilitas_cv.png", dpi=300, bbox_inches="tight")
plt.close(); print("[SAVED] mod2_05_stabilitas_cv.png")

# ── GAMBAR 6: Imbalance ──
fig6, axes6 = plt.subplots(1, 2, figsize=(12, 5))
fig6.patch.set_facecolor(BG)
fig6.suptitle("Analisis Distribusi Dataset\nCleveland (Artikel) vs Kaggle (Modifikasi 2)",
              fontweight="bold", color=RED_DARK, fontsize=13)
cle_c = [160,54,35,35,13]; cle_n = ["Healthy","Sick-Low","Sick-Med","Sick-High","Sick-Ser"]
b_a = axes6[0].bar(cle_n, cle_c, color=RED, alpha=0.85, edgecolor=RED_DARK)
axes6[0].set_title("Cleveland Dataset (297 sampel, 5 kelas)", fontweight="bold", fontsize=11)
axes6[0].set_facecolor(BG); axes6[0].spines[["top","right"]].set_visible(False)
axes6[0].grid(axis="y", alpha=0.3, linestyle="--"); axes6[0].tick_params(axis="x", rotation=15)
for bar,v in zip(b_a, cle_c):
    axes6[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                  str(v), ha="center", va="bottom", fontsize=9, fontweight="bold", color=RED_DARK)
kag_c = [np.sum(y==k) for k in CLASSES]
b_b = axes6[1].bar(CLASS_NAMES, kag_c, color=GREY, alpha=0.85, edgecolor="#2c3e50")
axes6[1].set_title(f"Kaggle Dataset ({len(y)} sampel, 2 kelas)", fontweight="bold", fontsize=11)
axes6[1].set_facecolor(BG); axes6[1].spines[["top","right"]].set_visible(False)
axes6[1].grid(axis="y", alpha=0.3, linestyle="--")
for bar,v in zip(b_b, kag_c):
    axes6[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                  str(v), ha="center", va="bottom", fontsize=11, fontweight="bold", color="#2c3e50")
plt.tight_layout()
plt.savefig("mod2_06_analisis_imbalance.png", dpi=300, bbox_inches="tight")
plt.close(); print("[SAVED] mod2_06_analisis_imbalance.png")

# Print CV results table
print("\n RINGKASAN:")
print(f"{'Method':<12} {'Acc':>8} {'CV Mean':>10} {'CV Std':>10}")
for name in METHOD_NAMES:
    print(f"  {name:<12} {results[name]['overall_acc']:>7.1f}% {results[name]['cv_mean']:>9.1f}% {results[name]['cv_std']:>9.1f}%")

# Save results dict for Mod 3 to reference
import json
mod2_acc_out = {k: results[k]["overall_acc"] for k in results}
with open("mod2_results_acc.json", "w") as f:
    json.dump(mod2_acc_out, f)

print("\n[DONE] Semua grafik Mod 2 tersimpan!")