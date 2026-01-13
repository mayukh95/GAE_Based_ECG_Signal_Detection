ï»¿<div align="center">
  <h1>å ECG Arrhythmia Classification using Class-Conditional Graph Autoencoders</h1>
  
  [![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
  [![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
  [![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
  [![Accuracy](https://img.shields.io/badge/Accuracy-96%2B%25-brightgreen.svg)]()
  
  **A deep learning pipeline for real-time ECG arrhythmia detection combining CNN, Graph Neural Networks, and conditional generation.**
  
  [Key Features](#-key-features) â¢ [Results](#-results) â¢ [Architecture](#ïž-architecture) â¢ [Installation](#ïž-installation) â¢ [Usage](#-usage) â¢ [Documentation](#-documentation)
</div>

---

## ¯ Project Overview

Built a **production-ready ECG classification system** achieving **96%+ accuracy** on MIT-BIH Arrhythmia Database using a novel Class-Conditional Graph Autoencoder architecture.

### à What Makes This Special?

| Feature | Description | Impact |
|---------|-------------|---------|
| **Graph Neural Networks** | Leverages relationships between similar heartbeats using k-NN graphs | 96% accuracy |
| **Class-Conditional Decoder** | Reconstructs waveforms with class-specific patterns | Better interpretability |
| **Joint Optimization** | Learns reconstruction + classification together | More robust features |
| **Real-time Inference** | < 10ms per prediction on CPU | Production-ready |

---

##  Key Features

â **Deep Learning Pipeline**: End-to-end training on 87K+ ECG samples  
â **Graph-Based Architecture**: k-NN graphs capture sample relationships  
â **Multi-Task Learning**: Simultaneous reconstruction + classification  
â **Interactive Visualizations**: t-SNE, ROC curves, confusion matrices  
â **Deployment Ready**: Exportable to ONNX for production use  
â **Comprehensive Documentation**: Step-by-step notebooks with detailed explanations  

---

## Ê Results

### Performance Metrics

| Metric | Value |
|--------|-------|
| **Overall Accuracy** | 96%+ |
| **Macro F1-Score** | 0.91+ |
| **Weighted F1-Score** | 0.96+ |
| **Inference Time** | < 10ms (CPU) |

**Per-Class Performance:**
- â Normal (N): 99%+ accuracy
- â Supraventricular (S): 94%+ accuracy  
- â Ventricular (V): 97%+ accuracy
- â Fusion (F): 89%+ accuracy
- â Unknown (Q): 92%+ accuracy

---

## × Architecture

**Pipeline:**
\`\`\`
ECG Waveform (187 samples)
    â
CNN Encoder (3 conv layers) â 128D features
    â
k-NN Graph Construction (k=5 neighbors)
    â
GCN Encoder (2 layers) â Graph-aware latent space
    â
    âââââââââââââââââââ¬âââââââââââââââââââ
    â  Classifier     â  Class-Cond.     â
    â  (predict)      â  Decoder         â
    â     â           â     â            â
    â  5 classes      â  Reconstruction  â
    âââââââââââââââââââŽâââââââââââââââââââ
\`\`\`

**Key Components:**
- **ECGEncoder**: 1D CNN with batch normalization and dropout
- **GCNEncoder**: 2-layer graph convolution network
- **ClassConditionedDecoder**: Deconvolutional network with class embeddings
- **SimpleClassifier**: Multi-layer perceptron for arrhythmia type prediction

---

## , Technical Highlights

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
Joint loss function balances three objectives:
\`\`\`
L_total = L_reconstruction + Î»â Ã L_classification + Î»â Ã L_graph
\`\`\`

---

## à Installation

### Prerequisites
- Python 3.8+
- CUDA 11.0+ (optional, for GPU acceleration)

### Quick Start

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt

# Download dataset (MIT-BIH)
python scripts/download_data.py
\`\`\`

---

## Ú Usage

### Python Scripts

**Training:**
\`\`\`python
from src.models import EnhancedJointECGModel
from src.data_loader import get_dataloaders
from src.train import train_model

# Load data
train_loader, val_loader = get_dataloaders(
    train_path='data/mitbih_train.csv',
    batch_size=512
)

# Initialize model
model = EnhancedJointECGModel(
    input_length=187,
    latent_dim=128,
    num_classes=5
)

# Train
train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=200,
    lr=0.001,
    device='cuda'
)
\`\`\`

**Inference:**
\`\`\`python
from src.inference import ECGPredictor

# Load trained model
predictor = ECGPredictor('models/best_model.pth')

# Predict on new ECG
prediction, confidence = predictor.predict(new_ecg_data)
print(f"Predicted class: {prediction} (confidence: {confidence:.2%})")

# Get reconstruction
reconstructed = predictor.reconstruct(new_ecg_data)
\`\`\`

### Option 3: Command Line Interface

\`\`\`bash
# Train model
python src/train.py --data data/mitbih_train.csv --epochs 200 --batch-size 512

# Run inference
python src/inference.py --model models/best_model.pth --input test_ecg.csv --output predictions.csv

# Evaluate model
python src/evaluate.py --model models/best_model.pth --test-data data/mitbih_test.csv
\`\`\`

---

## Ê Dataset

**MIT-BIH Arrhythmia Database**
- **Source**: [PhysioNet](https://physionet.org/content/mitdb/1.0.0/)
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

**Note**: Due to size limitations, the dataset is not included in this repository. Download it using the provided script or from PhysioNet.

---

##  Technical Skills Demonstrated

### Machine Learning & AI
- â Deep Learning (PyTorch)
- â Graph Neural Networks (PyTorch Geometric)
- â Convolutional Neural Networks (1D CNN)
- â Autoencoders & Representation Learning
- â Multi-Task Learning
- â Class Imbalance Handling

### Data Science
- â Exploratory Data Analysis (EDA)
- â Feature Engineering
- â Model Evaluation & Validation
- â Statistical Analysis
- â Data Visualization (Matplotlib, Plotly)
- â Time Series Analysis

---

## Ö Documentation

- **[Methodology](docs/METHODOLOGY.md)**: Detailed explanation of algorithms and mathematical foundations
- **[API Reference](docs/API.md)**: Code usage guide and function documentation
- **[Quick Start](QUICK_START.md)**: Get started in 5 minutes

---


---

##  Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

---

## Ä License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## O Acknowledgments

- **Dataset**: MIT-BIH Arrhythmia Database from [PhysioNet](https://physionet.org/)
- **Framework**: [PyTorch](https://pytorch.org/) and [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
- **Inspiration**: Research in medical AI and graph neural networks

---

## â­ Star This Repository

If you find this project useful, please consider giving it a star! It helps others discover this work and motivates continued development.

---

<div align="center">
  <sub>Built with â€ïž using PyTorch and Graph Neural Networks for Healthcare AI</sub>
</div>
