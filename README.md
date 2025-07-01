# 🧬 Breast Cancer Wisconsin (Original) Dataset — Analysis & Prediction

This repository presents a machine learning project using the **Breast Cancer Wisconsin (Original) Dataset** for the binary classification of breast tumors as **benign** or **malignant** based on cytological features.

> 📢 This project is created for **educational and research purposes**. It acknowledges the original authors and adheres to citation requirements provided by the dataset donors.

---

## 📌 About the Dataset

The **Breast Cancer Wisconsin (Original) Dataset** was obtained from:

- **Dr. William H. Wolberg**, University of Wisconsin Hospitals, Madison, Wisconsin, USA  
- **Donor**: Prof. Olvi L. Mangasarian (mangasarian@cs.wisc.edu)  
- **Received by**: David W. Aha (aha@cs.jhu.edu)  
- **Date Donated**: July 15, 1992  
- **Data Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(original))

### ✅ Dataset Characteristics

- **Instances**: 699
- **Attributes**: 10 numeric features + 1 class label
- **Missing Values**: 16 instances contain missing data (marked as '?')
- **Class Distribution**:
  - Benign: 458 (65.5%)
  - Malignant: 241 (34.5%)

### 🔢 Attribute Information

| #  | Feature Name                  | Domain      |
|----|-------------------------------|-------------|
| 1  | Clump Thickness               | 1 – 10      |
| 2  | Uniformity of Cell Size       | 1 – 10      |
| 3  | Uniformity of Cell Shape      | 1 – 10      |
| 4  | Marginal Adhesion             | 1 – 10      |
| 5  | Single Epithelial Cell Size   | 1 – 10      |
| 6  | Bare Nuclei                   | 1 – 10      |
| 7  | Bland Chromatin               | 1 – 10      |
| 8  | Normal Nucleoli               | 1 – 10      |
| 9  | Mitoses                       | 1 – 10      |
| 10 | Class (2 = Benign, 4 = Malignant) | Categorical |

---

## 🧠 Project Goals

- 🧹 Clean and preprocess the dataset
- 📊 Explore and visualize feature distributions
- 🔍 Handle missing values
- 🧪 Train and evaluate multiple classification models
- 🧾 Interpret model results (e.g., confusion matrix, precision, recall)

---

## 🚀 Getting Started

### Clone the Repository

```bash
git clone https://github.com/PolaOssama/breast-cancer-wisconsin.git
cd breast-cancer-wisconsin
