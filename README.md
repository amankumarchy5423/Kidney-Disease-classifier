# 🩺 Kidney Disease Classification using Deep Learning

> An end-to-end deep learning pipeline for classifying kidney CT images into **Normal, Cyst, Tumor, and Stone** categories using **MobileNetV2 Transfer Learning, Keras Tuner, and MLflow**.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red)
![Keras Tuner](https://img.shields.io/badge/Keras%20Tuner-Hyperparameter%20Optimization-purple)
![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-blue)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Motivation](#-motivation)
- [Problem Statement](#-problem-statement)
- [Objectives](#-objectives)
- [Features](#-features)
- [Dataset](#-dataset)
- [Technology Stack](#-technology-stack)
- [Project Architecture](#-project-architecture)
- [Model Architecture](#-model-architecture)
- [Transfer Learning](#-transfer-learning)
- [Hyperparameter Tuning](#-hyperparameter-tuning)
- [Project Structure](#-project-structure)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Dataset Setup](#-dataset-setup)
- [Configuration](#-configuration)
- [Running the Project](#-running-the-project)
- [Training Pipeline](#-training-pipeline)
- [Model Evaluation](#-model-evaluation)
- [MLflow Experiment Tracking](#-mlflow-experiment-tracking)
- [Output Artifacts](#-output-artifacts)
- [Results](#-results)
- [Performance Considerations](#-performance-considerations)
- [GPU Support](#-gpu-support)
- [Troubleshooting](#-troubleshooting)
- [Limitations](#-limitations)
- [Future Improvements](#-future-improvements)
- [Roadmap](#-roadmap)
- [Testing](#-testing)
- [Contributing](#-contributing)
- [License](#-license)
- [Medical Disclaimer](#-medical-disclaimer)
- [References](#-references)
- [Acknowledgements](#-acknowledgements)
- [Author](#-author)

---

# 📌 Overview

**Kidney Disease Classification** is a Deep Learning project designed to classify kidney CT images into four categories:

1. **Normal**
2. **Cyst**
3. **Tumor**
4. **Stone**

The project implements an end-to-end machine learning pipeline covering:

```text
Dataset
   ↓
Data Ingestion
   ↓
Image Preprocessing
   ↓
Model Building
   ↓
Hyperparameter Tuning
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Model Artifact