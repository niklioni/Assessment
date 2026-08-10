# Facial Expression Recognition — Classical Computer Vision Pipeline

A from-scratch pipeline for recognizing facial expressions (emotions) in images using **classical computer vision features** — no deep learning, no pretrained networks. Faces are located with a Haar Cascade detector, described with hand-engineered feature descriptors (HOG, LBP, Gabor filter banks), and classified with a linear SVM. The pipeline is evaluated and cross-validated on two standard facial-expression benchmarks, JAFFE and CK+.

The goal of the project was to explore how far purely classical feature engineering can get on a task that is nowadays almost always solved with CNNs — and to compare feature descriptors head-to-head under an identical modeling and evaluation setup.

## Pipeline overview

```mermaid
flowchart LR
    A[Raw images] --> B["Face detection\n(Haar Cascade)"]
    B --> C["Feature extraction\nHOG / LBP / Gabor"]
    C --> D["StandardScaler + PCA"]
    D --> E["Linear SVM\n(class-balanced)"]
    E --> F["Emotion label"]
```

1. **Face detection** — OpenCV's Haar Cascade classifier crops the face region out of each raw image.
2. **Feature extraction** — each cropped, resized face is turned into a fixed-length numeric vector using one of three classical descriptors (or a concatenation of two):
   - **HOG** (Histogram of Oriented Gradients) — captures edge/gradient structure, good at encoding overall face shape.
   - **LBP** (Local Binary Patterns, computed on an 8×8 grid of blocks) — captures local micro-texture patterns.
   - **Gabor filter bank** (2 scales × 4 wavelengths × 8 orientations, mean/variance pooled) — captures multi-scale, multi-orientation texture responses.
3. **Dimensionality reduction & scaling** — `StandardScaler` followed by `PCA`.
4. **Classification** — a class-balanced linear SVM (`LinearSVC`).
5. **Model selection** — a grid search over PCA components and SVM `C` is cross-validated per dataset (see below), and the best configuration is refit and evaluated on a held-out test split.

## Results

Six feature configurations were benchmarked on both datasets, each with its own cross-validated hyperparameter search. Numbers below are held-out test-set accuracy.

| Feature set   | Avg. CV accuracy | JAFFE test acc. | CK+ test acc. |
|---------------|:-----------------:|:----------------:|:---------------:|
| HOG           | 0.6745            | 0.6389           | 0.6783          |
| LBP           | 0.5835            | 0.4444           | 0.5664          |
| Gabor         | 0.4579            | 0.3333           | 0.3846          |
| HOG + Gabor   | **0.6822**        | 0.6389           | **0.6783**      |
| **HOG + LBP** | 0.6647            | **0.7222**       | 0.6783          |
| Gabor + LBP   | 0.5713            | 0.5000           | 0.5455          |

**Takeaways**

- HOG alone is consistently the strongest single descriptor — gradient/shape information generalizes better than pure texture for this task.
- Gabor features underperform on their own, but still add a small boost when combined with HOG.
- **HOG + LBP** gives the best result on JAFFE, suggesting shape and micro-texture cues are complementary once face crops are aligned.
- Combining descriptors doubles feature dimensionality but only helps marginally — with this small amount of training data, the classical pipeline is closer to its ceiling than a CNN would be.

Confusion matrices for the best-performing combination (HOG + LBP) are generated in the notebook for both datasets.

## Datasets

The pipeline was benchmarked on two well-known facial-expression datasets (not included in this repository — download them from their original sources and update the paths in the notebook):

- **JAFFE** (Japanese Female Facial Expression) — 7 expression classes, evaluated with a *person-disjoint* split (Leave-One-Subject-Out cross-validation via `GroupKFold`, grouped by subject ID parsed from the filename) so the model is never validated on a face it has seen during training.
- **CK+** (Extended Cohn-Kanade) — expression classes organized into folders (`train/`, `test/`, one subfolder per class), evaluated with 5-fold stratified cross-validation.

## Repository structure

```
.
├── preprocessing/
│   └── face_detection.py       # Haar Cascade face detection + dataset loaders (JAFFE, CK+)
├── features/
│   ├── hog_features.py         # HOG descriptor
│   ├── lbp_features.py         # Block-wise Local Binary Pattern histograms
│   └── gabor_features.py       # Multi-scale / multi-orientation Gabor filter bank
├── notebooks/
│   └── sample.ipynb            # End-to-end experiment: load data → extract features →
│                                #   grid-search + cross-validate → evaluate → confusion matrices
├── haarcascade_frontalface_default.xml   # OpenCV pretrained face detector
├── requirements.txt
└── README.md
```

## Getting started

```bash
git clone https://github.com/niklioni/Assessment.git
cd Assessment
pip install -r requirements.txt
```

Download JAFFE and/or CK+ into a local `data/` folder, then update the dataset paths at the top of `notebooks/sample.ipynb` to point at them. From there, run the notebook top to bottom — it will detect faces, build every feature configuration, grid-search + cross-validate a PCA + Linear SVM pipeline for each, and print/plot the results shown above.

## Tech stack

Python · OpenCV · scikit-learn · scikit-image · NumPy · Matplotlib · Seaborn

## Possible extensions

- Swap the Haar Cascade for a more robust detector (e.g. MTCNN/dlib) to reduce missed/false face crops.
- Add a CNN baseline (transfer learning) to quantify the gap between classical features and learned representations on the same splits.
- Feature selection / weighting instead of plain concatenation when combining descriptors.
