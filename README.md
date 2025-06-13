# 🫀 Graph Neural Networks for Heart Disease Prediction

## 📖 Introduction

Cardiovascular disease (CVD) remains the leading cause of mortality worldwide. Early prediction and diagnosis are essential for reducing the burden of heart conditions. Traditional diagnostic tools like stress testing and manual evaluation are resource-intensive and not always accessible. This project leverages **Graph Neural Networks (GNNs)** to model the relationships between patients using clinical features, offering a novel and efficient way to predict heart disease severity.

## 🧠 Objective

To build and compare **Graph Convolutional Networks (GCN)** and **Graph Attention Networks (GAT)** for heart disease classification using the UCI Heart Disease dataset. Each patient is modeled as a node in a graph, with edges representing similarity in medical history.

---

## 🧪 Methods

### 🔹 Node and Edge Construction

- **Nodes**: Represent individual patients, using clinical features (e.g., cholesterol, blood pressure).
- **Edges**: Represent similarity between patients, constructed using:
  1. **Manual rules** (based on thresholds of medical metrics)
  2. **Autoencoders** (for learned latent-space similarity)
  3. **K-Nearest Neighbors (KNN)** (best performance; K=5)

> **Note**: Dataset features were normalized to prevent skewed similarity due to varying feature scales.

### 🏷️ Label Masking

To avoid label leakage in the graph, we implemented **masked label aggregation**:
- During training, a node only accesses the labels of its **neighbors**, not its own.
- Achieved via adjacency matrix multiplication: `Y' = A × Y`.

---

## 🏥 Dataset

- **Source**: UCI Heart Disease Dataset (303 records)
- **Target**: Severity of heart disease (classes 0–4)
- **Features**: Age, cholesterol, blood pressure, max heart rate, etc.
- **Handling Missing Data**:
  - Numerical fields: Mean imputation
  - Categorical fields: Mode imputation

### ⚠️ Class Imbalance Handling

- **Stratified sampling** used for train/test split (80/20)
- **Weighted loss functions** to penalize minority class errors

---

## 🧰 Models

### 🔸 Previous Work (Baseline)
- Neural Network (MLP) with two hidden layers (64, 32 neurons)
- ~3,141 parameters
- Performance:
  - Precision: 0.47
  - Recall: 0.52
  - F1-score: 0.49
  - AUC-ROC: 0.775

### 🔸 Graph Neural Networks (This Project)

#### 1. **Graph Convolutional Network (GCN)**
- 2 GCN layers
- Graph normalization and dropout (p=0.3)
- ReLU activation
- Output layer: Softmax
- Parameters: **473**
  
#### 2. **Graph Attention Network (GAT)**
- 2 GAT layers with multi-head attention
- Dropout and graph norm
- Parameters: **515**
- Best performance with **weighted edges**

---

## 📊 Results

| Model           | Precision | Recall | F1-score | AUC-ROC |
|----------------|-----------|--------|----------|---------|
| MLP (Baseline) | 0.47      | 0.52   | 0.49     | 0.775   |
| **GCN**        | 0.65      | 0.66   | 0.65     | 0.810   |
| **GAT**        | **0.68**  | **0.69**| **0.68** | **0.832** |

> GAT with **KNN-based weighted edges** achieved the best trade-off between performance and model complexity.

---

## 🧠 Graph Construction Summary

| Method     | Description                                                  | Performance Impact        |
|------------|--------------------------------------------------------------|----------------------------|
| Manual     | Edges based on rules; inconsistent, lacked nuance            | ❌ Low                     |
| Autoencoder| Feature compression to infer similarity                      | ⚠️ Slight improvement      |
| **KNN**    | Distance-based similarity (K=5), unweighted and weighted     | ✅ Best results            |

---

## 📈 Visualization

- **KNN Graphs** show clusters of patients with similar disease severity.
- **Intermediate nodes** often bridge healthier and more severe groups.

---

## 🔮 Future Work

- **Larger datasets** to improve generalizability
- **Dynamic graph construction** based on evolving patient data
- **Temporal modeling** to account for disease progression
- **Hyperparameter tuning** (e.g., varying K in KNN)
- **Feature selection** for enhanced interpretability

---

## 🛠️ Tech Stack

- **Python**, **PyTorch**, **PyTorch Geometric**
- **scikit-learn** (for preprocessing, evaluation)
- **matplotlib** / **networkx** (visualization)

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/heart-disease-gnn.git
cd heart-disease-gnn
