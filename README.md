## Repository Contents

*   `base_knn_model.R` - Initial baseline k-NN model.
*   `knn_model.R` - Tuned k-NN model (optimizing for ROC AUC, tuning distance metrics).
*   `data/` - Contains training and testing datasets.
*   `plots/` - Visualizations of hyperparameter tuning results (ROC AUC vs Neighbors).

# MGMT Methylation Status Prediction in Glioblastoma (GBM)

This repository explores the feasibility of using radiomics-based machine learning models to predicting MGMT methylation status in Glioblastoma Multiforme (GBM) patients.

This work is part of the **URO REX Program at UBC** under the guidance of **Dr. Huan Zhong**.

## Reference Paper

This project attempts to replicate and build upon the findings from the following study:

**Improving MGMT methylation status prediction of glioblastoma through optimizing radiomics features using genetic algorithm-based machine learning approach**
*Duyen Thi Do, Ming-Ren Yang, Luu Ho Thanh Lam, Nguyen Quoc Khanh Le & Yu-Wei Wu*
*Scientific Reports* volume 12, Article number: 13412 (2022)
[Access the Article](https://www.nature.com/articles/s41598-022-17707-w)

The original paper achieved a model performance (Accuracy: 92.5%, Sensitivity: 0.894) using a Genetic Algorithm (GA)-based wrapper model on radiomics features extracted from MRI images from TCIA.

## Project Overview

### 1. Data and Preprocessing
Utilized radiomics features extracted from multimodal MRI images, downloaded from original author's github repository.
*   **Feature Selection**: Focused on the 20 key radiomics features identified in the reference paper (selected via XGBoost, figure 5).
*   **tidying**: Observations with missing values were removed.

### 2. Modeling Approach
Chose **k-Nearest Neighbors (k-NN)** as the baseline algorithm to establish a performance benchmark.

*   `base_knn_model.R`: A Straightforward implementation to test the data pipeline with hyperparameter tuning (number of neighbors `k`, weight function).
*   `knn_model.R`: Improvements including:
    *   **Metric Optimization**: Switched from Accuracy to **ROC AUC** for possible class imbalance.
    *   **Distance Metric**: Tuned `dist_power` to compare Manhattan (1) vs. Euclidean (2) distances (including 1.5).

### Results and Discussion

Initial results from `base_knn_model.R` suggested a promising testing accuracy of ~70.5%.

However, from `knn_model.R`, the model was largely predicting the majority class for most patients from the limited number of observations in the training + testing dataset. The inadequate ROC AUC score of 0.498 indicates that despite the high accuracy, the model could not effectively rank the observations correctly or distinguish the signal from noise using only these specific 20 features with a k-NN approach. 

## Future Improvements

Move to ensemble models like Random Forest/XGBoost, or SVM, while working with a larger, original dataset.