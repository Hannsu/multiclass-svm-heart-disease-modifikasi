import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

# ─── WARNA TEMA ────────────────────────────────────────────────────────────
RED   = "#C8192C"
CREAM = "#F5EDE4"
DARK  = "#2B2B2B"
LIGHT = "#F0B3B8"

# ─── 1. LOAD DATASET ─────────────────────────────────────────────────────────

print("=" * 60)
print("  MODIFIKASI 3 – DATASET SEDERHANA (dataheart_sederhana.csv)")
print("=" * 60)

df = pd.read_csv(
    r"C:\Users\ferdi\Downloads\tugas akhir ml teotzy\multiclass-svm-heart-disease-modifikasi\dataheart_sederhana.csv",
    sep=";"
)

print(f"\n[1] Dataset berhasil dimuat: {df.shape[0]} baris x {df.shape[1]} kolom")
print(df.head(3).to_string())

# ─── 2. EKSPLORASI DATASET ───────────────────────────────────────────────────
print(f"\n[2] Distribusi Target:")
print(df['target'].value_counts().rename({0: 'Sick (0)', 1: 'Healthy (1)'}))
print(f"\n[3] Missing Values: {df.isnull().sum().sum()}")
print(f"    Duplikat: {df.duplicated().sum()}")

# ─── 3. SPLIT DATA 80:20 ─────────────────────────────────────────────────────
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n[4] Pembagian Dataset 80:20")
print(f"    Training : {X_train.shape[0]} sampel")
print(f"    Testing  : {X_test.shape[0]} sampel")

# ─── 4. NORMALISASI ──────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ─── 5. TRAINING SVM ─────────────────────────────────────────────────────────
model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
model.fit(X_train_sc, y_train)
print(f"\n[5] Model SVM (kernel=RBF, C=1.0, gamma=scale) selesai training")

# ─── 6. EVALUASI ─────────────────────────────────────────────────────────────
y_pred = model.predict(X_test_sc)

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='weighted')
rec  = recall_score(y_test, y_pred, average='weighted')
f1   = f1_score(y_test, y_pred, average='weighted')

print(f"\n[6] Hasil Evaluasi:")
print(f"    Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
print(f"    Precision : {prec:.4f}")
print(f"    Recall    : {rec:.4f}")
print(f"    F1-Score  : {f1:.4f}")
print("\n    Classification Report:")
print(classification_report(y_test, y_pred,
      target_names=['Sick (0)', 'Healthy (1)']))

# Cross-validation
cv_scores = cross_val_score(model, scaler.transform(X), y, cv=5)
print(f"    Cross-Val (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ─── 7. VISUALISASI ──────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 14), facecolor=CREAM)
fig.suptitle("Modifikasi 3 – Klasifikasi Penyakit Jantung (Dataset Sederhana)",
             fontsize=18, fontweight='bold', color=RED, y=0.98)

gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# ── Plot 1: Distribusi Target ─────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
counts = df['target'].value_counts().sort_index()
bars = ax1.bar(['Sick (0)', 'Healthy (1)'], counts.values,
               color=[RED, LIGHT], edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, counts.values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             str(val), ha='center', va='bottom', fontweight='bold', color=DARK)
ax1.set_title("Distribusi Target", fontweight='bold', color=RED)
ax1.set_ylabel("Jumlah Sampel")
ax1.set_facecolor(CREAM)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# ── Plot 2: Pembagian Data Train/Test ─────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
sizes = [len(X_train), len(X_test)]
labels = [f'Train\n{len(X_train)} ({len(X_train)/len(df)*100:.0f}%)',
          f'Test\n{len(X_test)} ({len(X_test)/len(df)*100:.0f}%)']
ax2.pie(sizes, labels=labels, colors=[RED, LIGHT],
        autopct='%1.1f%%', startangle=90,
        textprops={'color': DARK, 'fontweight': 'bold'},
        wedgeprops={'edgecolor': 'white', 'linewidth': 2})
ax2.set_title("Pembagian Data 80:20", fontweight='bold', color=RED)

# ── Plot 3: Confusion Matrix ──────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
cm = confusion_matrix(y_test, y_pred)
cmap = sns.light_palette(RED, as_cmap=True)
sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=ax3,
            xticklabels=['Sick (0)', 'Healthy (1)'],
            yticklabels=['Sick (0)', 'Healthy (1)'],
            linewidths=1, linecolor='white', cbar=False,
            annot_kws={"size": 14, "weight": "bold"})
ax3.set_title("Confusion Matrix", fontweight='bold', color=RED)
ax3.set_xlabel("Predicted", color=DARK)
ax3.set_ylabel("Actual", color=DARK)

# ── Plot 4: Metrik Evaluasi ────────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 0])
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
values  = [acc, prec, rec, f1]
bars4 = ax4.barh(metrics, values, color=[RED, LIGHT, RED, LIGHT],
                 edgecolor='white', linewidth=1.5, height=0.5)
for bar, val in zip(bars4, values):
    ax4.text(val + 0.005, bar.get_y() + bar.get_height()/2,
             f'{val:.4f}', va='center', fontweight='bold', color=DARK, fontsize=10)
ax4.set_xlim(0, 1.12)
ax4.set_title("Metrik Evaluasi", fontweight='bold', color=RED)
ax4.set_facecolor(CREAM)
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
ax4.axvline(x=1.0, linestyle='--', color='gray', linewidth=0.8, alpha=0.6)

# ── Plot 5: Cross-Validation ──────────────────────────────────────────────────
ax5 = fig.add_subplot(gs[1, 1])
folds = [f"Fold {i+1}" for i in range(5)]
ax5.bar(folds, cv_scores, color=RED, alpha=0.85, edgecolor='white', linewidth=1.5)
ax5.axhline(y=cv_scores.mean(), color=DARK, linestyle='--', linewidth=1.5,
            label=f'Mean = {cv_scores.mean():.4f}')
for i, v in enumerate(cv_scores):
    ax5.text(i, v + 0.002, f'{v:.4f}', ha='center', fontsize=9,
             fontweight='bold', color=DARK)
ax5.set_ylim(0.7, 1.05)
ax5.set_title("Cross-Validation (5-Fold)", fontweight='bold', color=RED)
ax5.set_ylabel("Accuracy")
ax5.legend(fontsize=9)
ax5.set_facecolor(CREAM)
ax5.spines['top'].set_visible(False)
ax5.spines['right'].set_visible(False)

# ── Plot 6: Korelasi Fitur ────────────────────────────────────────────────────
ax6 = fig.add_subplot(gs[1, 2])
corr = df.corr()['target'].drop('target').abs().sort_values(ascending=True)
colors_feat = [RED if v > 0.3 else LIGHT for v in corr.values]
ax6.barh(corr.index, corr.values, color=colors_feat,
         edgecolor='white', linewidth=1)
ax6.set_title("Korelasi Fitur–Target", fontweight='bold', color=RED)
ax6.set_xlabel("|Korelasi|")
ax6.set_facecolor(CREAM)
ax6.spines['top'].set_visible(False)
ax6.spines['right'].set_visible(False)
ax6.axvline(x=0.3, linestyle='--', color='gray', linewidth=0.8)

# ── Plot 7: Distribusi Age per Kelas ─────────────────────────────────────────
ax7 = fig.add_subplot(gs[2, 0])
df[df['target'] == 1]['age'].plot.hist(ax=ax7, bins=12, color=RED, alpha=0.7,
                                       label='Healthy (1)', edgecolor='white')
df[df['target'] == 0]['age'].plot.hist(ax=ax7, bins=12, color=LIGHT, alpha=0.7,
                                       label='Sick (0)', edgecolor='white')
ax7.set_title("Distribusi Usia per Kelas", fontweight='bold', color=RED)
ax7.set_xlabel("Usia")
ax7.set_ylabel("Frekuensi")
ax7.legend()
ax7.set_facecolor(CREAM)
ax7.spines['top'].set_visible(False)
ax7.spines['right'].set_visible(False)

# ── Plot 8: Scatter age vs thalach ────────────────────────────────────────────
ax8 = fig.add_subplot(gs[2, 1])
for label, color, name in [(1, RED, 'Healthy (1)'), (0, LIGHT, 'Sick (0)')]:
    subset = df[df['target'] == label]
    ax8.scatter(subset['age'], subset['thalach'], c=color, label=name,
                alpha=0.7, edgecolor='white', linewidth=0.5, s=50)
ax8.set_title("Usia vs Max Heart Rate", fontweight='bold', color=RED)
ax8.set_xlabel("Usia")
ax8.set_ylabel("Max Heart Rate (thalach)")
ax8.legend()
ax8.set_facecolor(CREAM)
ax8.spines['top'].set_visible(False)
ax8.spines['right'].set_visible(False)

# ── Plot 9: Tabel Ringkasan ────────────────────────────────────────────────────
ax9 = fig.add_subplot(gs[2, 2])
ax9.axis('off')
table_data = [
    ["Metrik", "Nilai"],
    ["Total Data", f"{len(df)}"],
    ["Training",  f"{len(X_train)} (80%)"],
    ["Testing",   f"{len(X_test)} (20%)"],
    ["Accuracy",  f"{acc*100:.2f}%"],
    ["Precision", f"{prec*100:.2f}%"],
    ["Recall",    f"{rec*100:.2f}%"],
    ["F1-Score",  f"{f1*100:.2f}%"],
    ["CV Mean",   f"{cv_scores.mean()*100:.2f}%"],
]
tbl = ax9.table(cellText=table_data[1:], colLabels=table_data[0],
                loc='center', cellLoc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(11)
tbl.scale(1.2, 1.6)
for (row, col), cell in tbl.get_celld().items():
    cell.set_edgecolor('white')
    if row == 0:
        cell.set_facecolor(RED)
        cell.set_text_props(color='white', fontweight='bold')
    elif row % 2 == 0:
        cell.set_facecolor('#F0B3B8')
    else:
        cell.set_facecolor(CREAM)
ax9.set_title("Ringkasan Hasil", fontweight='bold', color=RED, pad=10)

plt.savefig("modifikasi3_hasil.png", dpi=150, bbox_inches='tight',
            facecolor=CREAM)
print("\n[7] Visualisasi disimpan: modifikasi3_hasil.png")
plt.show()