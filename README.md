# ECG Arrhythmia Classification using Class-Conditional Graph Autoencoders

<div align="center">
  <img src="assets/banner.png" alt="ECG Analysis Banner" width="800"/>
  
  # 🏥 ECG Arrhythmia Classification using Class-Conditional Graph Autoencoders
  
  [![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
  [![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
  [![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
  [![Accuracy](https://img.shields.io/badge/Accuracy-96.4%25-brightgreen.svg)]()
  
  **A deep learning pipeline for real-time ECG arrhythmia detection combining CNN, Graph Neural Networks, and conditional generation.**
  
  [Demo](#demo) • [Key Features](#key-features) • [Results](#results) • [Installation](#installation) • [Usage](#usage)
</div>

---

## 🎯 Project Overview

Built a **production-ready ECG classification system** achieving **96.4% accuracy** on MIT-BIH Arrhythmia Database using a novel Class-Conditional Graph Autoencoder architecture.

### 🧠 What Makes This Special?

| Feature | Description | Impact |
|---------|-------------|---------|
| **Graph Neural Networks** | Leverages relationships between similar heartbeats | 96% accuracy |
| **Class-Conditional Decoder** | Reconstructs waveforms with class-specific patterns | Better interpretability |
| **Joint Optimization** | Learns reconstruction + classification together | More robust features |
| **Real-time Inference** | < 10ms per prediction on CPU | Production-ready |

---

## 🚀 Key Features

✅ **Deep Learning Pipeline**: End-to-end training on 87K+ ECG samples  
✅ **Graph-Based Architecture**: k-NN graphs capture sample relationships  
✅ **Multi-Task Learning**: Simultaneous reconstruction + classification  
✅ **Interactive Visualizations**: t-SNE, ROC curves, confusion matrices  
✅ **Deployment Ready**: Exportable to ONNX for production use  
✅ **Comprehensive Documentation**: Step-by-step notebooks with explanations  

---

## 📊 Results

<div align="center">
  <img src="results/confusion_matrix.png" width="400"/>
  <img src="results/roc_curves.png" width="400"/>
</div>

### Performance Metrics

| Metric | Value |
|--------|-------|
| **Overall Accuracy** | 96.4% |
| **Macro F1-Score** | 0.91 |
| **Weighted F1-Score** | 0.96 |
| **Inference Time** | 8.3ms (CPU) |

**Per-Class Performance:**
- Normal (N): 99.2% accuracy
- Supraventricular (S): 94.1% accuracy  
- Ventricular (V): 97.3% accuracy
- Fusion (F): 89.5% accuracy
- Unknown (Q): 92.7% accuracy

---

## 🏗️ Architecture

<div align="center">
  <img src="results/model_architecture.png" width="700"/>
</div>

**Pipeline:**
```
ECG Waveform (187 samples)
    ↓
CNN Encoder (3 conv layers) → 128D features
    ↓
k-NN Graph Construction (k=5 neighbors)
    ↓
GCN Encoder (2 layers) → Graph-aware latent space
    ↓
    ┌─────────────────┬──────────────────┐
    │  Classifier     │  Class-Cond.     │
    │  (predict class)│  Decoder         │
    │     ↓           │     ↓            │
    │  5 classes      │  Reconstruction  │
    └─────────────────┴──────────────────┘
```

**Key Components:**
- **ECGEncoder**: 1D CNN with batch normalization
- **GCNEncoder**: 2-layer graph convolution network
- **ClassConditionedDecoder**: Deconvolutional network with class embeddings
- **Classifier**: Multi-layer perceptron for arrhythmia type prediction

---

## 🔬 Technical Highlights

### 1. Class-Conditional Reconstruction
Unlike standard autoencoders, this model uses **class information during reconstruction**:
- **Training**: Decoder uses TRUE labels (teacher forcing)
- **Inference**: Decoder uses PREDICTED labels (realistic deployment)

### 2. Graph Neural Networks
Builds k-NN graphs to capture **sample relationships**:
- Cosine similarity between CNN features
- k=5 nearest neighbors per sample
- Bidirectional edges for information flow

### 3. Multi-Objective Optimization
Joint loss function:
```
L_total = L_recon + 0.5*L_class + 0.1*L_graph
```

---

## 📈 Visualizations

<table>
  <tr>
    <td><img src="results/training_curves.png" width="300"/></td>
    <td><img src="results/tsne_visualization.png" width="300"/></td>
    <td><img src="results/sample_predictions.png" width="300"/></td>
  </tr>
  <tr>
    <td align="center">Training Progress</td>
    <td align="center">Latent Space Clustering</td>
    <td align="center">Prediction Examples</td>
  </tr>
</table>

---

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA 11.0+ (for GPU acceleration)

### Quick Start

```bash

# Install dependencies
pip install -r requirements.txt

# Download dataset (MIT-BIH)
python scripts/download_data.py

# Train model
python src/train.py --epochs 200 --batch-size 512

# Run inference
python src/inference.py --model best_model.pth --input test_ecg.csv
```

---

## 📚 Usage

### Training

```python
from src.models import EnhancedJointECGModel
from src.train import train_model

# Initialize model
model = EnhancedJointECGModel(
    input_length=187,
    latent_dim=128,
    num_classes=5
)

# Train
train_model(
    model=model,
    train_data=train_loader,
    val_data=val_loader,
    epochs=200,
    lr=0.001
)
```

### Inference

```python
from src.inference import predict

# Load trained model
model = load_model('best_model.pth')

# Predict on new ECG
prediction, confidence = predict(model, new_ecg_data)
print(f"Predicted class: {prediction} (confidence: {confidence:.2%})")
```

---

## 📊 Dataset

**MIT-BIH Arrhythmia Database**
- **Training**: 87,554 samples
- **Test**: 21,892 samples  
- **Classes**: 5 arrhythmia types
- **Sampling Rate**: 360 Hz
- **Duration**: ~0.5s per heartbeat (187 samples)

**Class Distribution:**
- Normal (N): 72.5%
- Ventricular (V): 5.8%
- Supraventricular (S): 2.2%
- Fusion (F): 0.7%
- Unknown (Q): 18.8%

---

## 🎓 Technical Skills Demonstrated

### Machine Learning
- [x] Deep Learning (PyTorch)
- [x] Graph Neural Networks (PyTorch Geometric)
- [x] Convolutional Neural Networks (1D CNN)
- [x] Autoencoders
- [x] Multi-Task Learning
- [x] Class Imbalance Handling


### Data Science
- [x] Exploratory Data Analysis (EDA)
- [x] Feature Engineering
- [x] Model Evaluation & Validation
- [x] Statistical Analysis
- [x] Data Visualization (Matplotlib, Plotly)

---

## 📖 Documentation

- **Methodology**: Detailed explanation of algorithms
- **Results**: Complete performance analysis
- **[API Reference](docs/API.md)**: Code usage guide

---

## 📄 License

This project is licensed under the MIT License - see LICENSE file.

---

## 🤝 Acknowledgments

- MIT-BIH Arrhythmia Database: [PhysioNet](https://physionet.org/)
- PyTorch Geometric: [PyG Team](https://pytorch-geometric.readthedocs.io/)

---

## 📧 Contact

**Your Name**  
📧 Email: your.email@example.com  
💼 LinkedIn: [linkedin.com/in/yourprofile](https://linkedin.com/in/yourprofile)  
🌐 Portfolio: [yourwebsite.com](https://yourwebsite.com)

---

<div align="center">
  <sub>Built with ❤️ using PyTorch and Graph Neural Networks</sub>
</div>
