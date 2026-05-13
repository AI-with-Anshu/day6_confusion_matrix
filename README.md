# Day 6 — Confusion Matrix Deep Dive 📊

## What I built today
Understood and implemented the **4 core components** of a Confusion
Matrix — TP, TN, FP, FN — using the Breast Cancer dataset.
Built a metrics dashboard showing Accuracy, Precision, Recall, F1 Score.

---

## The 4 Building Blocks

| Term | Full Name      | Meaning                          |
|------|---------------|----------------------------------|
| TP   | True Positive  | Sick patient — correctly detected |
| TN   | True Negative  | Healthy patient — correctly cleared|
| FP   | False Positive | Healthy — wrongly flagged as sick  |
| FN   | False Negative | Sick — wrongly cleared as healthy  |

> FN is the most dangerous in medical scenarios!

---

## Formulas

```
Accuracy  = (TP + TN) / Total
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1 Score  = 2 × (P × R) / (P + R)
```

---

## Results

| Metric    | Score |
|-----------|-------|
| Accuracy  | ~94%  |
| Precision | ~95%  |
| Recall    | ~96%  |
| F1 Score  | ~95%  |

---

## Output Figure
![Day 6 Results](https://github.com/AI-with-Anshu/day6_confusion_matrix/blob/main/day6_results.png)

---

## How to Run
```bash
pip install scikit-learn matplotlib numpy
python day6_confusion_matrix.py
```

---

## Dataset
**Breast Cancer Wisconsin Dataset** — built into scikit-learn
- 569 samples | 30 features | 2 classes (Malignant / Benign)

---
*Day 6 of 90 | M.Tech AI Student — Daily ML Journey*
  
