# 🔬 Breast Cancer Histopathology Classification System

[![MATLAB](https://img.shields.io/badge/MATLAB-R2020b+-blue.svg)](https://www.mathworks.com/products/matlab.html)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-BreakHis-orange.svg)](https://www.kaggle.com/datasets/ambarish/breakhis)

 

<div align="center">

<img src="media/Header.png" width="826" height="413">
 
 
 
</div>
<br/>

A comprehensive machine learning pipeline for automated classification of breast cancer histopathology images using traditional ML algorithms with advanced feature extraction and selection techniques.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Methodology](#methodology)
- [File Structure](#file-structure)
- [Citation](#citation)
- [License](#license)

---

## 🎯 Overview

This project implements an **8-step pattern recognition pipeline** for binary classification of breast histopathology images (benign vs. malignant) using the **BreakHis dataset**. The system emphasizes:

- **Clinical accuracy**: Minimizing false negatives (missed cancer diagnoses)
- **Comprehensive feature extraction**: Multi-modal approach (texture, morphology, intensity)
- **Multiple feature selection methods**: 6 different algorithms for robust feature ranking
- **Rigorous validation**: k-fold cross-validation with statistical significance testing
- **Complete visualization**: 13 diagnostic figures for performance analysis

### 🏆 Performance Highlights

| Model | Accuracy | Sensitivity | Specificity | AUC |
|-------|----------|-------------|-------------|-----|
| **Random Forest** | 71.1% | 61.9% | 80.4% | 0.771 |
| **XGBoost** | 68.6% | 60.8% | 76.3% | 0.754 |
| **SVM** | 62.9% | 59.8% | 66.0% | 0.667 |

*Note: Performance on BreakHis 100X magnification with 85/15 train/test split*

---

## ✨ Features

### 🔬 **Medical Image Processing**
- **10-stage preprocessing pipeline**:
  - Grayscale conversion, resizing, noise reduction
  - CLAHE (Contrast Limited Adaptive Histogram Equalization)
  - Multi-scale Gaussian filtering, unsharp masking
  - Morphological operations, bilateral filtering
  - Intensity normalization

### 🧬 **Multi-Modal Feature Extraction (2767 features)**

#### **1. Texture Features**
- **HOG (Histogram of Oriented Gradients)**: Edge directionality (16×16 cells)
- **LBP (Local Binary Patterns)**: Micro-texture descriptors (32×32 cells)
- **GLCM (Gray-Level Co-occurrence Matrix)**: Spatial texture (8 offsets)
- **Gabor Filter Bank**: Multi-scale, multi-orientation (3 wavelengths × 4 orientations)

#### **2. Morphological Features**
- Edge statistics (Sobel, Canny, LoG)
- Harris corner detection
- Skeleton length, Euler number
- Shape descriptors (area, perimeter, solidity, eccentricity, extent)

#### **3. Intensity Features**
- Statistical moments (mean, variance, skewness, kurtosis)
- Percentiles (10th, 50th, 90th)
- HSV color statistics (H&E staining variations)

### 🎯 **Feature Selection Methods (6 Algorithms)**

| Method | Type | Characteristics | Time Complexity |
|--------|------|-----------------|-----------------|
| **ReliefF** | Filter | Distance-weighted, k-NN based | O(n²d) |
| **F-Score (ANOVA)** | Filter | Univariate statistical testing | O(nd) |
| **RFE** | Wrapper | Recursive elimination with SVM | O(d² × iterations) |
| **Tree-based** | Embedded | Random Forest impurity | O(n log n × trees) |
| **LASSO** | Embedded | L1 regularization (sparsity) | O(nd² × iterations) |
| **PCA** | Transform | Linear dimensionality reduction | O(d³) |

### 🤖 **Machine Learning Models**

#### **1. Support Vector Machine (SVM)**
- Kernel: Radial Basis Function (RBF)
- Hyperparameters: BoxConstraint [1, 10, 100], KernelScale [0.5, 1, 5]
- Preprocessing: LDA projection (Mahalanobis-like distance)

#### **2. Random Forest (Ensemble)**
- Trees: 100-200, MinLeafSize: 1-5
- Out-of-bag error estimation
- Feature importance ranking

#### **3. XGBoost (Gradient Boosting)**
- Method: LogitBoost (MATLAB implementation)
- Cycles: 100-200, LearnRate: 0.1-0.2
- Adaptive learning with boosting

### 📊 **Validation & Evaluation**

- **5-fold stratified cross-validation** for hyperparameter tuning
- **Patient-disjoint split** (no patient overlap between train/test)
- **McNemar's test** for statistical significance between models
- **Bootstrap confidence intervals** (95% CI, n=100)
- **Comprehensive metrics**: Accuracy, Sensitivity, Specificity, Precision, F1, AUC

### 📈 **Visualization Suite (13 Figures)**

1. 10-Stage Preprocessing Pipeline
2. LDA 1D Projection (class separation)
3. ReliefF Feature Importance (top 30)
4. Feature Selection Algorithm Comparison
5. PCA Scree Plot & Cumulative Variance
6. Hyperparameter Tuning Curves (K vs CV accuracy)
7. SVM Performance (ROC + Confusion Matrix + Metrics)
8. Random Forest Performance
9. XGBoost Performance
10. Metrics Comparison (all models)
11. Combined ROC Curves
12. Confusion Breakdown (TN/FP/FN/TP)
13. Confusion Matrices (side-by-side)

---

## 📂 Dataset

### BreakHis (Breast Cancer Histopathological Database)

- **Source**: [Kaggle - BreakHis](https://www.kaggle.com/datasets/ambarish/breakhis)
- **Original Paper**: Spanhol et al. (2016) - IEEE TBME
- **Images**: 7,909 microscopic images (700 patients)
- **Magnifications**: 40X, 100X, 200X, 400X
- **Staining**: Hematoxylin & Eosin (H&E)

#### **Classes**

**Benign (2,480 images):**
- Adenosis
- Fibroadenoma
- Phyllodes tumor
- Tubular adenoma

**Malignant (5,429 images):**
- Ductal carcinoma
- Lobular carcinoma
- Mucinous carcinoma
- Papillary carcinoma

#### **Dataset Characteristics**
- RGB color images (PNG format)
- Resolution: Variable (typically 460×700 pixels)
- Tissue samples: Formalin-fixed, paraffin-embedded (FFPE)
- Ground truth: Expert pathologist diagnosis (double-verified)

---

## 🏗️ System Architecture

### 8-Step Pattern Recognition Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│ STEP 1: PROBLEM DEFINITION                                  │
│ - Binary classification (benign vs malignant)               │
│ - Clinical focus: Minimize false negatives                  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 2: DATA ACQUISITION                                    │
│ - Load BreakHis dataset (100X magnification)                │
│ - Patient-disjoint split: 85% train / 15% test             │
│ - Data leakage verification                                 │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 3: PREPROCESSING (10 stages)                           │
│ - Grayscale → Resize → Denoise → CLAHE → Filter            │
│ - Sharpen → Morphology → Bilateral → Normalize             │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 4: FEATURE EXTRACTION (2767 features)                  │
│ - HOG, LBP, GLCM, Gabor (texture)                          │
│ - Edge, corner, shape (morphology)                         │
│ - Moments, percentiles, HSV (intensity)                    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 5: FEATURE SELECTION (6 methods)                       │
│ - Variance/Correlation filtering → ~150 features           │
│ - ReliefF, F-Score, RFE, Tree, LASSO, PCA ranking          │
│ - Grid search: K ∈ {100, 150, 200, 250}                    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 6: MODEL SELECTION                                     │
│ - SVM (RBF kernel + LDA preprocessing)                     │
│ - Random Forest (ensemble of decision trees)               │
│ - XGBoost (gradient boosting)                              │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 7: TRAINING & HYPERPARAMETER TUNING                    │
│ - 5-fold stratified cross-validation                       │
│ - Grid search over hyperparameter space                    │
│ - Final training on full training set                      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 8: EVALUATION (held-out test set)                      │
│ - Confusion matrix analysis (TN/FP/FN/TP)                  │
│ - Metrics: Acc, Sens, Spec, Prec, F1, AUC                  │
│ - Statistical tests: McNemar, Bootstrap CI                  │
│ - Performance comparison with benchmarks                    │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Installation

### Prerequisites

- MATLAB R2020b or later
- Required Toolboxes:
  - Image Processing Toolbox
  - Statistics and Machine Learning Toolbox
  - Computer Vision Toolbox

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/breast-cancer-classification.git
   cd breast-cancer-classification
   ```

2. **Download BreakHis dataset**
   - Download from [Kaggle](https://www.kaggle.com/datasets/ambarish/breakhis)
   - Extract to `BreakHis_Main/` folder in project root

3. **Verify folder structure**
   ```
   breast-cancer-classification/
   ├── BreastCancer_Enhanced_Complete.m
   ├── BreakHis_Main/
   │   ├── benign/
   │   │   ├── adenosis/
   │   │   ├── fibroadenoma/
   │   │   ├── phyllodes_tumor/
   │   │   └── tubular_adenoma/
   │   └── malignant/
   │       ├── ductal_carcinoma/
   │       ├── lobular_carcinoma/
   │       ├── mucinous_carcinoma/
   │       └── papillary_carcinoma/
   └── README.md
   ```

---

## 💻 Usage

### Basic Usage

```matlab
% Run the complete pipeline
BreastCancer_Enhanced_Complete()
```

### Output

All results are saved to `Training_results/` folder:

```
Training_results/
├── models.mat              % Trained models (for deployment)
├── results.mat             % Complete results & analysis
├── training_log.txt        % Execution log
└── 01-13_*.png            % 13 visualization figures
```

### Loading Trained Models

```matlab
% Load models for prediction
load('Training_results/models.mat', 'trained');

% Access models
svmModel = trained.models.SVM;
rfModel = trained.models.RF;
xgbModel = trained.models.XGB;

% Access preprocessing parameters
ldaPreprocessor = trained.branchA.ldaModel;
zscoreParams = trained.branchB.zParams;
selectedFeatures = trained.featureIndices;
```

### Customization

Edit configuration in the code:

```matlab
% Dataset
cfg.magnification = '100X';  % Options: '40X', '100X', '200X', '400X'
cfg.prepTrainRatio = 0.85;   % Train/test split

% Feature selection
cfg.K_candidates = [100 150 200 250];
cfg.varianceThreshold = 0.005;
cfg.correlationThreshold = 0.90;

% Cross-validation
cfg.cvFolds = 5;

% Model hyperparameters
cfg.svm.boxConstraints = [1 10 100];
cfg.rf.numTrees = [100 200];
cfg.xgb.numCycles = [100 200];
```

---

## 📊 Results

### Performance Metrics (100X Magnification)

| Model | Accuracy | Sensitivity | Specificity | Precision | F1 | AUC | False Negatives |
|-------|----------|-------------|-------------|-----------|----|----|-----------------|
| **Random Forest** | **71.1%** | **61.9%** | 80.4% | 75.9% | 0.682 | **0.771** | **37 / 97** |
| XGBoost | 68.6% | 60.8% | 76.3% | 72.0% | 0.659 | 0.754 | 38 / 97 |
| SVM | 62.9% | 59.8% | 66.0% | 63.7% | 0.617 | 0.667 | 39 / 97 |

### Confusion Matrix (Random Forest - Best Model)

```
                Predicted
                Benign  Malignant
Actual  Benign    78       19       (Specificity: 80.4%)
        Malignant 37       60       (Sensitivity: 61.9%)
```

**Clinical Interpretation:**
- **True Positives (60)**: Correctly identified malignant cases
- **False Negatives (37)**: Missed cancers ⚠️ CRITICAL
- **False Positives (19)**: Unnecessary anxiety/procedures
- **True Negatives (78)**: Correctly identified benign cases

### Statistical Significance

**McNemar's Test (p-values):**
- SVM vs Random Forest: p = 0.0375* (significant)
- SVM vs XGBoost: p = 0.1531 (not significant)
- RF vs XGBoost: p = 0.4990 (not significant)

**Bootstrap 95% Confidence Intervals:**
- Random Forest: 70.7% [63.4%, 76.3%]
- XGBoost: 68.5% [60.8%, 75.3%]
- SVM: 63.1% [56.7%, 70.1%]

### Comparison with State-of-the-Art

| Method | Accuracy | Sensitivity | AUC | Year |
|--------|----------|-------------|-----|------|
| **Proposed (RF)** | **71.1%** | **61.9%** | **0.771** | 2024 |
| Spanhol et al. | 84.6% | 82.0% | 0.850 | 2016 |
| Gupta & Bhavsar | 88.0% | 86.5% | 0.890 | 2017 |
| Araújo et al. | 83.4% | 81.0% | 0.840 | 2017 |

*Note: Performance gap due to single magnification (100X) vs. multi-scale approaches in benchmarks*

---

## 🔬 Methodology

### Feature Selection Comparison

| Algorithm | Computation Time | Top 20 Agreement | Advantages |
|-----------|-----------------|------------------|------------|
| ReliefF | 2.5s | Baseline | Distance-weighted, robust |
| F-Score | 0.04s | 65% overlap | Fast, statistical |
| RFE | 220s | 70% overlap | Iterative, accurate |
| Tree-based | 6.3s | 60% overlap | Non-linear interactions |
| LASSO | 0.2s | 75% overlap | Sparse solution |
| PCA | 0.04s | N/A | Linear transform |

### Hyperparameter Tuning Results

**Grid Search Results (5-Fold CV):**

| K Features | SVM CV Acc | RF CV Acc | XGB CV Acc |
|------------|------------|-----------|------------|
| 100 | 84.7% | 82.8% | **85.1%** |
| 150 | **85.5%** | 82.8% | 83.9% |
| 151 | **85.5%** | 82.7% | 84.3% |

**Best Configuration:** K = 151 features, CV Accuracy = 85.5%

**Note:** Gap between CV (85.5%) and test (71.1%) indicates overfitting - future work should address this.

---

## 📁 File Structure

```
breast-cancer-classification/
│
├── BreastCancer_Enhanced_Complete.m   # Main pipeline script
├── README.md                           # This file
├── LICENSE                             # MIT License
│
├── BreakHis_Main/                     # Raw dataset (user downloads)
│   ├── benign/
│   └── malignant/
│
├── BreakHis/                          # Prepared dataset (auto-generated)
│   ├── Training/
│   │   ├── benign/
│   │   └── malignant/
│   └── Test/
│       ├── benign/
│       └── malignant/
│
└── Training_results/                  # Output folder (auto-generated)
    ├── models.mat                     # Trained models
    ├── results.mat                    # Complete results
    ├── training_log.txt               # Execution log
    └── 01-13_*.png                   # Visualization figures
```

---

## 🎓 Key Concepts

### Clinical Focus: Minimizing False Negatives

In cancer diagnosis, **False Negatives are critical errors**:
- Missed cancer → Delayed treatment → Poor prognosis
- Target: **High Sensitivity** (>85% ideal)
- Current: 61.9% sensitivity = 38% miss rate ⚠️

**Trade-off:**
- Increasing sensitivity → More false positives → Unnecessary biopsies
- Balance required: Maximize sensitivity while maintaining acceptable specificity

### Patient-Disjoint Split

**Why it matters:**
- **Problem**: Same patient's images in train & test → Data leakage
- **Solution**: Split by patient ID, not random images
- **Result**: Realistic performance estimation

**Implementation:**
```matlab
% Extract patient IDs from filenames
trainPatients = unique(extractPatientID(trainFiles));
testPatients = unique(extractPatientID(testFiles));

% Verify no overlap
if ~isempty(intersect(trainPatients, testPatients))
    error('Patient leakage detected!');
end
```

### Distance Metrics in Classification

| Model | Distance Metric | How it Works |
|-------|----------------|--------------|
| **SVM** | Mahalanobis-like (implicit) | RBF kernel + LDA preprocessing |
| **Random Forest** | Not distance-based | Threshold-based splits on features |
| **XGBoost** | Not distance-based | Gradient boosting with weak learners |

---

## 🐛 Known Issues & Limitations

### Current Limitations

1. **Performance below benchmarks** (71% vs 85%)
   - **Cause**: Single magnification (100X) vs multi-scale in papers
   - **Solution**: Combine multiple magnifications

2. **High false negative rate** (38%)
   - **Cause**: Small training set (1078 images)
   - **Solution**: Data augmentation or deep learning

3. **Overfitting** (CV 85.5% → Test 71%)
   - **Cause**: Small test set (194 images), limited generalization
   - **Solution**: More diverse training data

4. **PCA yields only 1 component**
   - **Cause**: Missing standardization before PCA
   - **Status**: Fixed in latest version

### Future Improvements

- [ ] Multi-magnification ensemble (40X + 100X + 200X + 400X)
- [ ] Deep learning models (CNN: ResNet, VGG, DenseNet)
- [ ] Data augmentation (rotation, flip, color jitter)
- [ ] Transfer learning from ImageNet pre-trained models
- [ ] Attention mechanisms for interpretability
- [ ] Deployment as web application or mobile app

---

## 📚 References

### Dataset

1. **Spanhol, F. A., Oliveira, L. S., Petitjean, C., & Heutte, L. (2016)**  
   "A Dataset for Breast Cancer Histopathological Image Classification"  
   *IEEE Transactions on Biomedical Engineering*, 63(7), 1455-1462.  
   [DOI: 10.1109/TBME.2015.2496264](https://doi.org/10.1109/TBME.2015.2496264)

### Benchmarks

2. **Gupta, V., & Bhavsar, A. (2017)**  
   "Breast Cancer Histopathological Image Classification: Is Magnification Important?"  
   *IEEE Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)*

3. **Araújo, T., et al. (2017)**  
   "Classification of Breast Cancer Histology Images Using Convolutional Neural Networks"  
   *PLoS ONE*, 12(6), e0177544.

### Methods

4. **Kononenko, I. (1994)**  
   "Estimating Attributes: Analysis and Extensions of RELIEF"  
   *European Conference on Machine Learning*

5. **Tibshirani, R. (1996)**  
   "Regression Shrinkage and Selection via the LASSO"  
   *Journal of the Royal Statistical Society*, Series B, 58(1), 267-288.

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Areas for Contribution

- Implementing deep learning models (CNN)
- Adding more feature extraction methods
- Improving preprocessing pipeline
- Creating GUI for easy usage
- Writing unit tests
- Documentation improvements

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**BreakHis Dataset License:**
The BreakHis dataset is publicly available for research purposes. Please cite the original paper (Spanhol et al., 2016) when using the dataset.

---

## 👨‍💻 Author

**Dr. Erna**  
Expert in Machine Learning & Medical Image Processing

### Contact
- Email: your.email@example.com
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your Name](https://linkedin.com/in/yourprofile)

---

## 🙏 Acknowledgments

- **BreakHis Dataset**: Spanhol et al. for providing the publicly available dataset
- **MATLAB**: MathWorks for excellent image processing and ML toolboxes
- **Medical Domain Experts**: For insights on clinical significance and validation

---

## 📊 Project Status

🟡 **Status**: Active Development  
🎯 **Goal**: Achieve >85% accuracy & sensitivity  
📅 **Last Updated**: December 2024  
🔄 **Version**: 1.0.0

---

## 🔗 Related Projects

- [Deep Learning for Histopathology](https://github.com/user/deep-histo)
- [Medical Image Segmentation](https://github.com/user/med-seg)
- [Cancer Detection Toolkit](https://github.com/user/cancer-toolkit)

---

**⭐ If you find this project useful, please consider giving it a star!**

**📧 Questions? Open an issue or contact the author.**
