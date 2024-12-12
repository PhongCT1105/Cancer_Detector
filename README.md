# Predicting Cancer Type and Identifying Key Cancer Genes Using Genetic Sequencing

## Project Overview
This project leverages machine learning to predict cancer presence and type based on gene expression levels and to identify key genes implicated in cancer development. By combining supervised and unsupervised techniques, the system provides accurate predictions and valuable insights for early detection and personalized treatment planning.

## Key Features
- Predicts cancer presence and specific cancer types using gene expression levels.
- Identifies key genes involved in cancer development.
- Provides an interactive demo for researchers and clinicians.
- Incorporates state-of-the-art clustering techniques for gene analysis.

## Dataset
The project uses the CuMiDa dataset, a curated repository of microarray sequencing data from cancerous and healthy tissues. The dataset includes:
- 78 datasets representing 13 cancer types.
- Gene expression data from 12 to 357 samples per dataset.
- Over 54,000 features per dataset.

The dataset is accessible at [CuMiDa repository](https://sbcb.inf.ufrgs.br/cumida).

## Motivation
Advancements in gene sequencing make it possible to classify cancers and identify shared genetic pathways. This project aims to:
- Enable personalized treatment planning.
- Facilitate early cancer detection.
- Support biotechnology and pharmaceutical research.

## Methodology
1. **Data Processing and EDA:**
   - Used GPL 570 data with gene expression for 12 cancers and healthy tissue.
   - Cleaned and pre-processed datasets for analysis.

2. **Classification Tasks:**
   - Cancer type prediction using Logistic Regression and Random Forest.
   - Subtype classification with PCA and Random Forest.
   - 5-fold cross-validation for model evaluation.

3. **Clustering:**
   - Reduced dimensionality with PCA.
   - Performed K-means clustering to analyze cancer similarities and gene relationships.

## Results
- **Classification:**
  - Baseline accuracy: 88% for cancer type prediction.
  - Final model accuracy: 90% with 100% recall for all cancer types.
- **Key Genes:**
  - Identified important genes for each cancer type.
  - Overlap analysis of feature selection methods revealed robust gene markers.
- **Clustering:**
  - Demonstrated clear cancer type differentiation with PCA and clustering metrics.

## Access and Demo
- Project code: [GitHub Repository](https://github.com/CS539Team3)
- Interactive demo: [Streamlit App](https://genes2cancer.streamlit.app/)

## Conclusion
This project demonstrates the potential of machine learning in cancer diagnostics and gene analysis, offering tools for researchers and clinicians to advance cancer treatment and understanding.

## References
For detailed references and methodologies, consult the accompanying [Final Report](./Final%20Report.pdf).
