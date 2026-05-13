import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import ( accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, 
                             classification_report )

# ── Step 1: Dataset load ──────────────────────────
# Breast Cancer dataset — 2 classes: Malignant(bura) / Benign(theek)
data = load_breast_cancer()
X = data.data        # 30 features — measurements of tumor
y = data.target      # 0 = Malignant, 1 = Benign

print(f"Total samples : {X.shape[0]}")
print(f"Total features: {X.shape[1]}")
print(f"Classes       : {data.target_names}")

# ── Step 2: Train-Test Split ───────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# ── Step 3: training Model ───────────────────────────
model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X_train, y_train)

# ── Step 4: Prediction ──────────────────────────────
y_pred = model.predict(X_test)

# ── Step 5: Confusion Matrix values  ─────────────
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()  # ravel = flatten karo

print(f"\n{'='*40}")
print(f"True Positive  (TP) : {tp}  ← Yes, Cancer")
print(f"True Negative  (TN) : {tn}  ← No, Cancer")
print(f"False Positive (FP) : {fp}   ← No, Cancer but predicted")
print(f"False Negative (FN) : {fn}   ← Yes, Cancer but missed")
print(f"{'='*40}")

# ── Step 6: Metrics calculation  ────────────────────
acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec  = recall_score(y_test, y_pred)
f1   = f1_score(y_test, y_pred)

print(f"\nAccuracy  : {acc:.4f}  ({acc*100:.2f}%)")
print(f"Precision : {prec:.4f}")
print(f"Recall    : {rec:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"\n{classification_report(y_test, y_pred, target_names=data.target_names)}")

# ── Step 7: Create Figure ───────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1 — Confusion Matrix
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=data.target_names)
disp.plot(ax=axes[0], colorbar=False)
axes[0].set_title("Confusion Matrix\n(Breast Cancer Dataset)")

# Plot 2 — Metrics bar chart
metrics      = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
values       = [acc, prec, rec, f1]
colors       = ['#378ADD', '#1D9E75', '#D85A30', '#7F77DD']
bars = axes[1].bar(metrics, values, color=colors, width=0.5)
axes[1].set_ylim(0, 1.1)
axes[1].set_title("Model Evaluation Metrics")
axes[1].set_ylabel("Score")

# print value over every bar
for bar, val in zip(bars, values):
    axes[1].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.02,
        f"{val:.2f}", ha='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig("day6_results.png", dpi=150)
plt.show()
print("Figure saved — day6_results.png")
