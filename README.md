# 🧬 Predicting Cancer Type and Identifying Key Cancer Genes Using Genetic Sequencing

## 🌟 Project Overview
This project harnesses the power of **machine learning** to:
- 🩺 **Detect Cancer**: Predict whether a patient has cancer and classify its type.
- 🔬 **Identify Key Genes**: Highlight significant genes involved in cancer development.
- 📊 **Enhance Insights**: Provide tools for personalized treatment and early detection.

---

## 📂 Dataset
We used the **CuMiDa** dataset, a curated repository of gene expression data for cancerous and healthy tissues:
- 📋 **78 datasets** from over 30,000 studies.
- 🧬 Gene expression for **13 cancer types** (e.g., pancreatic, breast, liver, brain).
- 🧪 Up to **54,676 features** per dataset.
- **Access the dataset**: [CuMiDa repository](https://sbcb.inf.ufrgs.br/cumida)

---

## 🚀 Motivation
With advancements in **gene sequencing**, this project addresses:
- **Personalized Treatment**: Tailor therapies based on genetic profiles.
- **Early Detection**: Improve success rates by identifying cancers early.
- **Biotechnology Research**: Unlock shared pathways for disease progression.

---

## 🛠️ Methodology

### **1. Data Processing and EDA** 🔄
- Cleaned datasets and selected **GPL 570** data for 12 cancer types + healthy tissue.
- Preprocessed for clarity and feature selection.

### **2. Classification Tasks** 🎯
- Cancer Type Prediction:
  - Logistic Regression & Random Forest with **88%-90% accuracy**.
- Cancer Subtype Prediction:
  - PCA + Random Forest for refined **subtype accuracy**.

### **3. Clustering** 🔗
- Dimensionality reduction using **PCA**.
- Applied **K-means clustering** to reveal cancer similarities and key genetic insights.

### 📈 **Visualization Pipeline**:
![Pipeline](https://github.com/PhongCT1105/Cancer_Detector/raw/main/Cancer_Detection.drawio.png)

---

## 📊 Results

### ✅ **Cancer Classification**
- **Baseline Accuracy**: 88%
- **Final Model**:
  - Accuracy: **90%**
  - Recall: **100%**

### 🧬 **Key Gene Identification**
- Top genes for each cancer type were identified.
- Overlapping feature selection methods highlighted robust gene markers.

### 🔍 **Clustering**
- Demonstrated cancer differentiation using PCA and K-means.
- Metrics:
  - Adjusted Rand Index (ARI)
  - Silhouette Score

---

## 🌐 Access and Demo
- **📂 Code**: [GitHub Repository](https://github.com/PhongCT1105/Cancer_Detector)
- **🌟 Live Demo**: [Streamlit App](https://genes2cancer.streamlit.app/)

---

## 📣 Conclusion
This project:
- Advances cancer diagnostics through machine learning.
- Offers **actionable insights** for early detection and personalized treatments.
- Supports **clinical research** with robust models and feature analyses.

---

## 📚 References
For more details, consult:
- [CuMiDa dataset documentation](https://sbcb.inf.ufrgs.br/cumida)
- [Project Report](./Final%20Report.pdf)
