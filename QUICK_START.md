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
- 📓 [Training Notebook](notebooks/training-ecg.ipynb)

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

## 📚 Explore the Notebooks (5-10 minutes)

```bash
# Start Jupyter
jupyter notebook

# Open:
# 1. notebooks/training-ecg.ipynb - See full training pipeline
# 2. notebooks/detection-ecg.ipynb - See inference & evaluation
```

Both notebooks have:
- ✅ Step-by-step explanations
- ✅ Visualizations at every stage
- ✅ Beginner-friendly comments
- ✅ Real examples

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

---

## 💼 For Your Resume/CV

**One-Line:**
> ECG Classification using Graph Neural Networks (96%+ accuracy) | PyTorch | [GitHub Link]

**Detailed:**
> Developed production-ready ECG arrhythmia classification system using 
> Graph Neural Networks, achieving 96.4% accuracy on 87K+ samples.
> Implemented novel class-conditional autoencoder architecture combining
> CNN + GNN for multi-task learning. Built complete pipeline from data
> processing to deployment-ready inference (<10ms latency).
>
> Tech: PyTorch, PyTorch Geometric, Scikit-learn, Python
> Link: github.com/YOUR_USERNAME/ECG-Arrhythmia-Classification

---

## 🔍 What to Look For

### Code Quality
- ✅ Modular design (`src/` package)
- ✅ Clear function names and docstrings
- ✅ Type hints where appropriate
- ✅ Consistent style

### Documentation
- ✅ Mathematical foundations explained
- ✅ Architecture diagrams
- ✅ Usage examples
- ✅ Performance analysis

### Engineering Practices
- ✅ Version control ready (.gitignore)
- ✅ Dependencies managed (requirements.txt)
- ✅ Reproducible experiments
- ✅ Command-line tools

---

## ❓ Common Questions

**Q: Can I run this without GPU?**
> Yes! Inference takes ~8ms on CPU. Training will be slower but works.

**Q: Do I need the full dataset?**
> No! Sample data is included. Full dataset instructions in README.

**Q: Is this suitable for production?**
> Yes! Export to ONNX, wrap in FastAPI, deploy in Docker.

**Q: What ML concepts does this demonstrate?**
> Deep learning, CNNs, Graph Neural Networks, autoencoders, 
> multi-task learning, class imbalance handling, time series.

---

## 📞 Next Steps

1. ⭐ **Star the repo** (if you found it useful!)
2. 📖 **Read README.md** for complete overview
3. 💻 **Try the notebooks** for interactive experience
4. �� **Check docs/** for technical details

---

## �� That's It!

You now understand this project. 

**For interviews, be ready to discuss:**
- Why Graph Neural Networks for ECG
- Class-conditional reconstruction approach
- Handling 113:1 class imbalance
- Production deployment strategy

**Time to explore:** 5 minutes
**Time to master:** 1-2 hours
**Career impact:** Significant 🚀

---

[Back to Main README](README.md) | [Upload to GitHub](GITHUB_UPLOAD_GUIDE.md)
