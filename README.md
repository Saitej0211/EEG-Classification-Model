
# Seizure and Non-Seizure Classification

## Overview

This project aims to develop a classification model for Electroencephalogram (EEG) data to diagnose epilepsy. The focus is on creating a patient-specific detector for identifying the onset of seizures using machine learning techniques. The datasets used include the CHB-MIT EEG Database and the Bonn EEG Dataset.

## Table of Contents
1. [Introduction](#introduction)
2. [Strategy](#strategy)
3. [Analysis](#analysis)
4. [Results](#results)
5. [Conclusion](#conclusion)
6. [Limitation](#limitation)
7. [Future Work](#future-work)

## Introduction

Epilepsy is a neurological disorder characterized by recurrent seizures. This project develops a model to classify EEG data to detect seizures early. The EEG recordings from the CHB-MIT and Bonn EEG datasets are used for training and evaluation. The goal is to improve the accuracy and timeliness of seizure detection to enhance patient care.

## Strategy

The project employs a supervised machine learning approach to build a patient-specific seizure detector. Key challenges include:
- Overlapping EEG patterns between seizure and non-seizure states.
- Variability in EEG signals across different patients.
- The need for rapid and accurate detection with limited seizure samples.

The chosen approach involves:
- Using supervised classification to distinguish between seizure and non-seizure periods.
- Employing a Long Short-Term Memory (LSTM) model for its effectiveness in capturing temporal dependencies in sequential data.

## Analysis

### Data Preprocessing

1. **Data Acquisition:** Download and extract the CHB-MIT EEG dataset.
2. **Preprocessing:** Resample EEG signals to 100,000 Hz and normalize the data.
3. **Data Organization:** Convert the data into a Pandas DataFrame, separating signals and labels.

### Feature Extraction

- Extract time-domain and frequency-domain features from EEG signals.
- Use 100,000 time-domain samples for analysis.

### Data Splitting

- Divide the data into training, validation, and test sets to ensure model robustness and prevent overfitting.

### Model Selection

- **Model:** Long Short-Term Memory (LSTM) network.
- **Parameters:** 
  - `input_size`: Number of features.
  - `hidden_size`: Number of hidden units.
  - `num_layers`: Number of LSTM layers.
  - `batch_first=True`: Batch dimension first.

### Model Training

- Train the LSTM model using techniques like dropout and early stopping to prevent overfitting.
- Utilize ReLU activation function and SoftMax for output layer transformations.

## Results

- **Evaluation Metrics:** Accuracy, precision, recall, and F1-score.
- **Model Performance:** The LSTM model demonstrated effective classification of EEG data, with enhanced performance through hyperparameter tuning.

## Conclusion

The project successfully developed an LSTM-based model for EEG classification, capable of identifying seizures with improved accuracy. Data preprocessing and feature extraction were effectively managed, and model training strategies ensured generalization to unseen data.

## Limitation

- **Dataset Diversity:** Limited dataset size and diversity may affect the model's applicability to broader populations.
- **Hardware Dependencies:** Training requires significant computing power, particularly for deep learning models.
- **Model Interpretability:** LSTM models may be challenging to interpret, which could impact clinical application.

## Future Work

- **Model Architecture:** Explore alternative neural network architectures such as CNNs and hybrid models.
- **Data Augmentation:** Investigate methods to augment the dataset and improve model robustness.
- **Real-Time Processing:** Develop models for real-time EEG data analysis to enhance clinical decision-making.
