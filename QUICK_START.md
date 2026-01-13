# ⚡ Quick Start Guide

Get started with the ECG Arrhythmia Classification project in 5 minutes!

---

## 🎯 For Recruiters/Reviewers

**TL;DR:** This project demonstrates production-ready deep learning for healthcare AI.

- **What:** ECG arrhythmia classification using Graph Neural Networks
- **Accuracy:** 96.4% on MIT-BIH database (87K+ samples)
- **Innovation:** Class-conditional graph autoencoders (novel architecture)
- **Stack:** PyTorch, PyTorch Geometric, Python
- **Deployment:** Real-time inference (<10ms on CPU)

**Quick Links:**
- 📖 [Full Documentation](README.md)
- 🔬 [Technical Methodology](docs/METHODOLOGY.md)
- 📊 [Performance Results](docs/RESULTS.md)
- 💻 [API Reference](docs/API.md)

---

## 🚀 Run the Demo (1 minute)

```bash
# Install dependencies
pip install -r requirements.txt

# Generate sample data
python scripts/download_data.py --create-sample

# Test the model (Python)
python -c "
from src.models import EnhancedJointECGModel
import torch

model = EnhancedJointECGModel()
sample = torch.randn(1, 1, 187)
edge_index = torch.LongTensor([[0], [0]])

z, recon, logits, pred = model.forward_inference(sample, edge_index)
print(f'Prediction: Class {pred.item()}')
print('✓ Model works!')
"
```

---

## 🔧 Use the Code (Production)

### Training

```python
from src.train import train_model

train_model(
    train_path='data/mitbih_train.csv',
    epochs=200,
    batch_size=512,
    device='cuda'
)
```

### Inference

```python
from src.inference import ECGPredictor

predictor = ECGPredictor('models/best_model.pth')
prediction, confidence = predictor.predict(ecg_waveform)

print(f"Prediction: {prediction} ({confidence:.1%} confidence)")
```

### Command Line

```bash
# Train
python src/train.py --train-data data/train.csv --epochs 200

# Predict
python src/inference.py --model models/best_model.pth --input data/test.csv
```

---

## 📖 Read the Documentation

### For Technical Deep Dive:
1. [METHODOLOGY.md](docs/METHODOLOGY.md) - Architecture, math, algorithms
2. [RESULTS.md](docs/RESULTS.md) - Performance analysis, benchmarks
3. [API.md](docs/API.md) - Complete code reference

### For Quick Reference:
- [README.md](README.md) - Overview, features, installation
- [notebooks/](notebooks/) - Interactive examples

---

## 🎯 Key Features to Highlight

### Innovation
- ✨ **Novel Architecture:** Class-conditional graph autoencoders
- 🔗 **Graph Neural Networks:** Captures relationships between samples
- 🎨 **Multi-Task Learning:** Classification + reconstruction

### Performance
- 📊 **96.4% Accuracy** on 87K+ samples
- ⚡ **<10ms Inference** on CPU
- 🎯 **Handles Imbalance:** 113:1 class ratio

### Production-Ready
- 🏭 **Clean Architecture:** Modular, reusable code
- 📝 **Comprehensive Docs:** Math, API, examples
- 🧪 **Tested & Validated:** Multiple evaluation metrics
- 🚀 **Deployment-Ready:** ONNX export, CLI tools


[Back to Main README](README.md) | [Upload to GitHub](GITHUB_UPLOAD_GUIDE.md)
