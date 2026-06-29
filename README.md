# Project 2: Data Classification Using AI

## Overview
This repository contains the implementation of **Project 2 (Data Classification)** as part of the AI Engineering Internship at **DecodeLabs**. This project transitions from rule-based programming into **Supervised Learning** by building a fundamental machine learning pipeline using algorithmic logic.

The primary objective is to classify the benchmark **Iris Dataset** using the **K-Nearest Neighbors (KNN)** algorithm.

## Project Pipeline (Architecture)

The system strictly follows the standard IPO (Input-Process-Output) framework:

1. **Input:** Loading the benchmark Iris dataset (150 balanced samples, 4 dimensions, 3 classes).
2. **Process:**
   * **Data Shuffling & Splitting:** Dividing the dataset into an 80% training set and a 20% testing set.
   * **Feature Scaling:** Applying `StandardScaler` (Mean = 0, Variance = 1) to eliminate dimensional bias.
   * **Model Training:** Training a K-Nearest Neighbors Classifier with an optimal $K=5$ value.

3. **Output:** Evaluating the classification decisions using a Confusion Matrix and the macro-average F1 Score to avoid the "Accuracy Mirage".

## Prerequisites & Installation

Ensure you have Python installed, then set up the required dependencies via the terminal:
```bash
pip install scikit-learn pandas numpy