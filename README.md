# Outlier Detection in IoT using Trained Autoencoder and Contrastive Loss

## Overview
This repository provides supplementary materials and a reference implementation for the research paper entitled:

**“Outlier Detection in IoT using Trained Autoencoder and Contrastive Loss”**

The paper introduces an unsupervised outlier detection framework designed for Internet of Things (IoT) environments. The proposed method combines a trained autoencoder with contrastive loss to learn discriminative latent representations, enabling effective separation between normal and anomalous data without requiring labeled samples.

---

## Method Description
The proposed approach consists of:
- A **trained autoencoder** that encodes high-dimensional IoT data into a compact latent space and reconstructs the original input.
- A **contrastive loss mechanism** that enforces similarity among normal data points while increasing separation from anomalous samples in the latent space.
- A **reconstruction-error-based decision rule**, where data points with reconstruction errors exceeding a defined threshold are classified as outliers.

The integration of contrastive learning enhances the discriminative capability of the autoencoder and improves robustness in complex and heterogeneous IoT data.

---

## Datasets
Experiments in the paper are conducted on two publicly available benchmark datasets commonly used in IoT and anomaly detection research.

### 1. Statlog (Landsat Satellite)
- **Number of Instances:** 6,435  
- **Number of Features:** 36  
- **Domain:** Environmental monitoring and climate-related IoT data  
- **Description:**  
  This dataset contains multispectral satellite imagery features. Outliers correspond to abnormal spectral patterns, making it suitable for evaluating unsupervised outlier detection in environmental IoT scenarios.
- **Source:**  
  https://archive.ics.uci.edu/dataset/146/statlog+landsat+satellite

### 2. UNSW-NB15
- **Number of Instances:** 175,341  
- **Number of Features:** 49  
- **Domain:** IoT and network security  
- **Description:**  
  The UNSW-NB15 dataset includes realistic network traffic with normal behavior and diverse attack types. In this work, anomalous traffic patterns and intrusion events are treated as outliers, providing a representative benchmark for IoT security applications.
- **Source:**  
  https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15

---

## Experimental Setup
- Input features are normalized before training.
- No synthetic anomalies are generated; the model detects naturally occurring outliers in the datasets.
- Evaluation metrics include **Recall, Precision, Accuracy, and F1-Score**.
- Baseline methods for comparison include DBSCAN, k-Means, One-Class SVM, Bayesian Networks, and a standard Autoencoder.

---

## Reproducibility
A sample implementation of the proposed method is provided for reproducibility and research purposes. Experimental settings and parameter configurations follow the descriptions presented in the paper.

---

## License
This repository is intended for academic and research use. Users must comply with the original licenses of the Statlog and UNSW-NB15 datasets when using the data.

