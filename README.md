# Handwritten Digit Recognition – Multi-Layer Perceptron (From Scratch)

This repository contains the implementation, training, and evaluation of **Multi-Layer Perceptron (MLP)** architectures built completely from scratch using Python and `NumPy`. The project deliberately avoids high-level deep learning frameworks (such as PyTorch or TensorFlow) for the core neural network layers and backpropagation, focusing instead on first-principles understanding and optimization.

The target task is multi-class classification on the MNIST / Kaggle Digit Recognizer dataset ($28 \times 28$ grayscale images across 10 digit classes, $0$ through $9$).

---

## Table of Contents

1. [Architectural Overview](#architectural-overview)
2. [Data Processing Pipeline](#data-processing-pipeline)
3. [Mathematical Foundations](#mathematical-foundations)
4. [Models & Variants](#models--variants)
   - [Vanilla Full-Batch MLP](#1-vanilla-full-batch-mlp)
   - [Mini-Batch Gradient Descent](#2-mini-batch-mlp)
   - [Planned & Implemented Extensions](#3-planned--implemented-extensions)
5. [Evaluation & Diagnostics](#evaluation--diagnostics)
6. [Dependencies & Setup](#dependencies--setup)
7. [Usage](#usage)

---

## Architectural Overview

The baseline network configuration is structured as follows:

* **Input Layer:** 784 neurons (flattened $28 \times 28$ pixel vectors).
* **Hidden Layer:** 100 neurons with **Sigmoid** activation.
* **Output Layer:** 10 neurons with numerically stable **Softmax** activation (producing normalized probabilities for digits 0–9).
* **Loss Criterion:** **Categorical Cross-Entropy** (bounded with epsilon clipping to prevent numerical underflow/overflow).
* **Weight Initialization:** Xavier/He-scaled Gaussian initialization:
  $$W \sim \mathcal{N}\left(0, \sqrt{\frac{1}{n_{\text{in}}}}\right)$$

---

## Data Processing Pipeline

* **Dataset:** Kaggle Digit Recognizer (`train.csv`).
* **Feature Scaling:** Pixel values normalized from $[0, 255]$ to the range $[0, 1]$ via float division by $255.0$.
* **Stratified Data Partitioning:**
  * **Train Set:** 70%
  * **Validation Set:** 15%
  * **Test Set:** 15%
  * *Stratification* guarantees balanced digit class distributions across all three splits.
* **Target Encoding:** Class labels are one-hot encoded for loss calculation and gradient backpropagation.

---

## Mathematical Foundations

### 1. Activations
* **Sigmoid:**
  $$\sigma(z) = \frac{1}{1 + e^{-z}}, \quad \sigma'(z) = \sigma(z)(1 - \sigma(z))$$

* **Numerically Stable Softmax:**
  $$\text{Softmax}(z_i) = \frac{e^{z_i - \max(z)}}{\sum_{j=1}^K e^{z_j - \max(z)}}$$

### 2. Loss Function
* **Categorical Cross-Entropy:**
  $$\mathcal{L} = -\frac{1}{m} \sum_{i=1}^m \sum_{k=1}^K y_{ik} \log(\hat{y}_{ik} + \epsilon)$$

### 3. Backpropagation (Analytical Gradients)
Given error $\delta^{[2]} = \hat{y} - y$ at the output layer:
$$\frac{\partial \mathcal{L}}{\partial W^{[2]}} = \frac{1}{m} (\delta^{[2]})^T A^{[1]}, \quad \frac{\partial \mathcal{L}}{\partial b^{[2]}} = \frac{1}{m} \sum_{i=1}^m \delta^{[2]}_i$$

Propagating to the hidden layer:
$$\delta^{[1]} = (\delta^{[2]} W^{[2]}) \odot \sigma'(Z^{[1]})$$
$$\frac{\partial \mathcal{L}}{\partial W^{[1]}} = \frac{1}{m} (\delta^{[1]})^T X, \quad \frac{\partial \mathcal{L}}{\partial b^{[1]}} = \frac{1}{m} \sum_{i=1}^m \delta^{[1]}_i$$

---

## Models & Variants

### 1. Vanilla Full-Batch MLP
* Computes gradients across the entire training partition per epoch.
* Records epoch-by-epoch loss and accuracy across Train, Validation, and Test sets.
* Integrates **Early Stopping** based on validation loss divergence with configurable `patience` and `min_delta`.

### 2. Mini-Batch MLP
* Subclasses the base implementation to perform mini-batch stochastic gradient descent (e.g., `batch_size = 64`).
* Improves gradient variance, accelerates convergence speed, and scales better to larger datasets.

### 3. Planned & Implemented Extensions
* **Regularization:**
  * L2 Regularization (Weight Decay)
  * Inverted Dropout on hidden activations
* **Advanced Optimizers:**
  * Momentum SGD
  * RMSprop / Adam
* **Data Augmentation:** Real-time affine transforms (rotations, translations, zoom, contour filters) using OpenCV and Keras preprocessing utilities.
* **Hyperparameter Tuning:** Automated search with Optuna (Bayesian optimization) and Stratified 5-Fold Cross-Validation.

---

## Evaluation & Diagnostics

The project provides comprehensive plotting utilities to monitor and inspect model behavior:

1. **Class Balance Distributions:** Bar charts verifying equal representation in Train, Validation, and Test partitions.
2. **Learning Curves (Accuracy):** Epoch-level accuracy trajectories comparing Train, Val, and Test performance.
3. **Loss Curves:** Categorical cross-entropy progression plotted on a logarithmic scale.
4. **Confusion Matrix:** Breakdown of inter-class confusion patterns (e.g., distinguishing between 4 and 9 or 3 and 8).
5. **Misclassification Inspection:** Visual grid of incorrectly classified digit samples alongside true vs. predicted labels.

---

## Dependencies & Setup

Ensure you have a Python 3.9+ environment installed. Install all necessary dependencies via `pip`:

```bash
pip install numpy pandas matplotlib scikit-learn opencv-python optuna tensorflow
```

---

## Usage

1. Place `train.csv` into the expected directory (default Kaggle path or a local `data/` folder).
2. Open and run the Jupyter notebook:
   ```bash
   jupyter notebook digit_recognizer.ipynb
   ```
3. Initialize and train the model directly in Python:

```python
from models import MiniBatchVanillaMLP

# Instantiate model
mlp = MiniBatchVanillaMLP(
    input_size=784,
    hidden_size=100,
    output_size=10,
    learning_rate=0.5,
    batch_size=64
)

# Train with early stopping monitoring
history = mlp.train_with_history(
    x_train, y_train_one_hot, y_train_labels,
    x_val, y_val_one_hot, y_val_labels,
    x_test, y_test_one_hot, y_test_labels,
    epochs=25,
    patience=10,
    min_delta=1e-4
)
```
