# 🚢 Underwater Vessel Noise Classification — ShipsEar Dataset

> **PVR Lab Course Project** | School of Computer Science | UPES Dehradun

A complete machine learning pipeline for classifying underwater vessel noise using the [ShipsEar dataset](https://drive.google.com/drive/folders/1PrO_ZBiYM9hmMQIdgqt7otH0886pO7Xr), covering handcrafted acoustic features, classical classifiers, and deep learning models.

---

## 👥 Author
Somya Pratap Singh 

## 📊 Dataset

**ShipsEar** — `shipsear_5s_16k`

📁 **Dataset Link (segmented):** [Google Drive](https://drive.google.com/drive/folders/1PrO_ZBiYM9hmMQIdgqt7otH0886pO7Xr)

| Class | Vessel Type | Segments |
|-------|-------------|----------|
| 0 (A) | Motorboats | 295 |
| 1 (B) | Mussel Boats | 241 |
| 2 (C) | Fishing Vessels | 674 |
| 3 (D) | Passengers / Ferries | — |
| 4 (E) | Ocean Liners / Tugboats | 179 |

Pre-segmented into **5-second clips at 16 kHz**. Original recordings at 52 kHz by Santos-Domínguez et al. (2016).

---

## 🏗️ Project Structure

```
ShipsEar_Project/
├── Part2_Data_Acquisition_EDA.ipynb        # EDA & integrity checks
├── Part3_Preprocessing_Splits.ipynb        # Preprocessing & splits
├── Part3_Splits_Fix_v2.ipynb               # Fix for path remapping
├── Part4_Feature_Extraction.ipynb          # Handcrafted + deep features
├── Part5_Classification_Models.ipynb       # KNN, SVM, RF, CNN, ResNet-18
├── Part5_CNN_Training_Updated.ipynb        # Updated 200-epoch training
├── Part6_Evaluation.ipynb                  # Metrics, ablation, error analysis
├── features/                               # Extracted .npy arrays
├── models/                                 # Saved .pkl and .pth files
├── Report/
│   ├── shipsear_report_final.pdf           # IEEE conference report
│   └── shipsear_report_final.tex           # LaTeX source
├── README.md
└── README.txt
```

---

## ⚙️ Dependencies

```bash
pip install librosa soundfile resampy soxr tqdm Pillow \
            scikit-learn numpy pandas matplotlib seaborn scipy \
            torch torchvision
```

> ⚠️ **On Google Colab:** Do **NOT** run `pip install torch` — it overwrites the pre-installed GPU version. Only install the other packages.

| Package | Version |
|---------|---------|
| Python | 3.12 |
| PyTorch | 2.10.0+cu128 |
| CUDA | 12.8 |
| librosa | ≥ 0.10.0 |
| scikit-learn | ≥ 1.4.0 |
| numpy | ≥ 1.26.0 |
| scipy | ≥ 1.11.0 |

---

## 🚀 How to Run

### Step 0 — Setup (Google Colab)

1. Go to [Google Colab](https://colab.research.google.com)
2. Enable GPU: **Runtime → Change runtime type → T4 GPU → Save**
3. Mount Drive and download dataset:

```python
from google.colab import drive
drive.mount('/content/drive')
```

4. Install missing packages:

```bash
!pip install librosa soundfile resampy soxr tqdm Pillow scikit-learn
```

---

### Step 1 — Exploratory Data Analysis

**Notebook:** `Part2_Data_Acquisition_EDA.ipynb`

- Update `DATASET_ROOT` to your `shipsear_5s_16k/` path
- Run all cells
- **Outputs:** `dataset_manifest.csv` + 8 figures

---

### Step 2 — Preprocessing & Splits

**Notebook:** `Part3_Preprocessing_Splits.ipynb`

- Update `DATASET_ROOT` and `PROCESSED_ROOT`
- Run all cells
- If `splits.json` paths don't resolve, run `Part3_Splits_Fix_v2.ipynb`
- **Outputs:** `shipsear_processed/` + `splits.json`

---

### Step 3 — Feature Extraction

**Notebook:** `Part4_Feature_Extraction.ipynb`

- Update `DATASET_ROOT` in Cell 0
- If paths in splits.json are from a different machine, run the path remapping cell first
- **Runtime:** ~5 min on T4 GPU
- **Outputs:** `features/` folder with `.npy` arrays and `scaler.pkl`

| Feature Group | Dimensions | Method |
|---------------|-----------|--------|
| MFCC + Δ + ΔΔ | 240 | 40 coeffs × 3 derivatives × (mean+std) |
| STFT bands | 40 | 20 frequency bands × (mean+std) |
| LOFAR peaks | 20 | Top-10 Welch PSD peaks |
| DEMON peaks | 10 | Top-5 blade-rate envelope peaks |
| **Total** | **310** | Concatenated, StandardScaler normalised |

Log-mel spectrograms: `(N, 1, 128, 128)` tensors at 128 mel bands.

---

### Step 4 — Classification Models

**Notebooks:** `Part5_Classification_Models.ipynb` + `Part5_CNN_Training_Updated.ipynb`

Run cells in order:

```
Cell 0  → Config (check DEVICE prints 'cuda')
Cell 1  → Load features
Cells 2-6   → KNN (k∈{1,3,5,7}), SVM (grid search), Random Forest
Cells 7-12  → CNN (200 epochs) + ResNet-18 (200 epochs)
Cells 13-16 → GradCAM, confusion matrices, ROC curves, results table
```

| Model | Architecture |
|-------|-------------|
| KNN | k ∈ {1,3,5,7}, Euclidean |
| SVM | RBF kernel, C∈{0.1,1,10,100}, γ∈{scale,auto,0.01,0.001} |
| Random Forest | n_estimators ∈ {100,200,500} |
| Custom CNN | 3× [Conv→BN→ReLU→MaxPool] → GAP → Dense(256) → Dense(5) |
| ResNet-18 | ImageNet pretrained, FC → Dense(5), all layers fine-tuned |

---

### Step 5 — Evaluation & Ablation

**Notebook:** `Part6_Evaluation.ipynb`

> ⚠️ Run **Cell 1b (resplit fix)** if class 3 is missing from training set.

- **Outputs:** `final_results_table.csv`, `ablation_results.csv`, `literature_comparison.csv` + 6 figures

---

## 📈 Results

### Final Results (5-seed mean ± std)

| Model | Acc (mean±std) | Macro-F1 | AUC |
|-------|---------------|----------|-----|
| KNN | 0.8485 ± 0.0052 | 0.8450 ± 0.0058 | 0.9082 ± 0.0034 |
| **SVM RBF** | **0.9084 ± 0.0015** | **0.9046 ± 0.0016** | **0.9903 ± 0.0005** |
| Random Forest | 0.9030 ± 0.0122 | 0.9020 ± 0.0118 | 0.9916 ± 0.0019 |
| CNN (scratch) | 0.8247 ± 0.0891 | 0.8351 ± 0.0796 | 0.9716 ± 0.0261 |
| ResNet-18 | 0.9020 ± 0.0135 | 0.8994 ± 0.0117 | 0.9879 ± 0.0016 |

> Single-run SVM-RBF test accuracy: **92.51%** — exceeds published SOTA (87.2%)

### Ablation Study (Random Forest, Val Accuracy)

| Feature Config | Dim | Val Acc |
|---------------|-----|---------|
| DEMON only | 10 | 0.451 |
| LOFAR only | 20 | 0.730 |
| MFCC only | 240 | 0.769 |
| STFT only | 40 | 0.841 |
| **All combined** | **310** | **0.868** |

### Literature Comparison

| Method | Reference | Accuracy |
|--------|-----------|----------|
| GMM + MFCC | Santos-Domínguez et al. (2016) | 0.726 |
| SVM + MFCC | Santos-Domínguez et al. (2016) | 0.754 |
| CNN + Mel | Irfan et al. (2021) | 0.843 |
| ResNet + Aug | Xie et al. (2022) | 0.872 |
| **Our SVM RBF** | **This work** | **0.925** ✅ |
| **Our Random Forest** | **This work** | **0.892** ✅ |

---

## 🖥️ Hardware Configuration

| Component | Specification |
|-----------|--------------|
| Platform | Google Colaboratory |
| GPU | NVIDIA Tesla T4 (16 GB VRAM) |
| CPU | Intel Xeon (2 vCPU) |
| RAM | 12 GB |
| PyTorch | 2.10.0+cu128 |
| CUDA | 12.8 |

---

## 🐛 Known Issues & Fixes

<details>
<summary><b>ModuleNotFoundError: No module named 'resampy'</b></summary>

```bash
!pip install resampy soxr
```
Or change `res_type='kaiser_best'` to `res_type='soxr_hq'` in the code.
</details>

<details>
<summary><b>TypeError: ReduceLROnPlateau got unexpected keyword 'verbose'</b></summary>

Remove `verbose=False` from the `ReduceLROnPlateau` call. Removed in PyTorch ≥ 2.4.
</details>

<details>
<summary><b>splits.json has 0 paths / Train = 0 segments</b></summary>

Run `Part3_Splits_Fix_v2.ipynb`. The list files contain absolute paths from the original machine (`E:\MTQP\...`). The fix remaps them to your local `DATASET_ROOT` automatically.
</details>

<details>
<summary><b>val_loss = nan during CNN training</b></summary>

Run the val-set resplit cell — all segments of one class ended up in val only. The fix does a stratified resplit of train+val arrays.
</details>

<details>
<summary><b>CNN/ResNet-18 low single-run accuracy (0.25–0.33)</b></summary>

Class 3 (Passengers/Ferries) was absent from the base training split. Use the 5-seed mean results which use proper stratified splits: CNN=0.825, ResNet-18=0.902.
</details>

---

## 📄 Report

The full IEEE conference format report is available at:
- `Report/shipsear_report_final.pdf` — compiled PDF
- `Report/shipsear_report_final.tex` — LaTeX source (for Overleaf)

---

## 📚 References

1. Santos-Domínguez et al., "ShipsEar: An underwater vessel noise database," *Applied Acoustics*, vol. 113, 2016.
2. Irfan et al., "DeepShip: An underwater acoustic benchmark dataset," *Expert Systems with Applications*, vol. 183, 2021.
3. Xie et al., "Underwater target recognition using CNN with data augmentation," *Applied Acoustics*, vol. 193, 2022.
4. He et al., "Deep residual learning for image recognition," *CVPR*, 2016.
5. Park et al., "SpecAugment," *Interspeech*, 2019.

---

<p align="center">
  Made with ❤️ at UPES Dehradun &nbsp;|&nbsp; PVR Lab 2026
</p>
