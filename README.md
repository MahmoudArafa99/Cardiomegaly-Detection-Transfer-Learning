# Cardiomegaly Detection using Transfer Learning

## 📝 Project Overview
Cardiomegaly, characterized by an enlarged heart, is a critical clinical finding often detected via chest X-rays. This project aims to automate the detection process using Deep Convolutional Neural Networks (CNNs). Given that medical images are complex and unstructured, Deep Learning is exceptionally well-suited for this task.

## 🛠️ Methodology: Transfer Learning Approach
The project implements a **Transfer Learning** strategy using the **DenseNet121** architecture, pre-trained on the **ImageNet** dataset. 

### Model Architecture & Customizations:
- **Base Model:** DenseNet121 (Pre-trained).
- **GlobalAveragePooling2D:** For dimensionality reduction.
- **Batch Normalization:** To stabilize training and accelerate convergence.
- **Dense Layers:** Custom classification head with ReLU activation.
- **Dropout (0.3):** Implemented for regularization to prevent overfitting.
- **Sigmoid Activation:** Final output for binary classification (Normal vs. Cardiomegaly).

### Training Strategy:
1. **Phase 1 (Feature Extraction):** The base model weights were frozen, and only the custom head was trained.
2. **Phase 2 (Fine-Tuning):** A selection of the final layers in the base model were unfrozen and re-compiled with a significantly reduced learning rate to optimize performance for medical X-ray features.

## 🚀 Optimization & Regularization
- **Data Augmentation:** Applied to increase data diversity.
- **EarlyStopping:** To prevent overfitting.
- **ReduceLROnPlateau:** To decrease learning rate when validation loss stagnates.

## 📊 Results and Evaluation
The fine-tuned model achieved a balanced performance:
- **Accuracy:** ~70%
- **Evaluation:** Rigorously tested using a Confusion Matrix and Classification Report to ensure high sensitivity in detecting Cardiomegaly cases.

## 📁 Dataset
The dataset used in this project is sourced from Kaggle:
[Cardiomegaly Disease Prediction Dataset](https://www.kaggle.com/datasets/rahimanshu/cardiomegaly-disease-prediction-using-cnn)
