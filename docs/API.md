# API Reference

Complete code usage guide for the ECG Classification package.

## Table of Contents
- [Models](#models)
- [Data Loading](#data-loading)
- [Graph Utilities](#graph-utilities)
- [Training](#training)
- [Inference](#inference)

---

## Models

### EnhancedJointECGModel

Main model class combining all components.

```python
from src.models import EnhancedJointECGModel

model = EnhancedJointECGModel(
    input_length=187,      # ECG waveform length
    latent_dim=128,        # Latent space dimension
    num_classes=5,         # Number of arrhythmia classes
    dropout_rate=0.3       # Dropout probability
)
```

#### Methods

**forward(waveforms, edge_index, class_labels)**
```python
# Training mode (uses true labels)
z, reconstructed, logits = model(waveforms, edge_index, true_labels)

# Arguments:
#   waveforms: (batch_size, 1, 187) - Input ECG waveforms
#   edge_index: (2, num_edges) - Graph connectivity
#   class_labels: (batch_size,) - True class labels
#
# Returns:
#   z: (batch_size, 128) - Latent representations
#   reconstructed: (batch_size, 1, 187) - Reconstructed waveforms
#   logits: (batch_size, 5) - Classification logits
```

**forward_inference(waveforms, edge_index)**
```python
# Inference mode (uses predicted labels)
z, reconstructed, logits, predicted = model.forward_inference(waveforms, edge_index)

# Returns:
#   z: (batch_size, 128) - Latent representations
#   reconstructed: (batch_size, 1, 187) - Reconstructed waveforms
#   logits: (batch_size, 5) - Classification logits
#   predicted: (batch_size,) - Predicted class labels
```

**encode(waveforms, edge_index)**
```python
# Extract latent representations only
z = model.encode(waveforms, edge_index)  # (batch_size, 128)
```

**predict(waveforms, edge_index)**
```python
# Get class probabilities
probs = model.predict(waveforms, edge_index)  # (batch_size, 5)
```

---

### ECGEncoder

CNN-based feature extractor.

```python
from src.models import ECGEncoder

encoder = ECGEncoder(
    input_length=187,
    latent_dim=128,
    dropout_rate=0.3
)

# Extract features
features = encoder(waveforms)  # (batch_size, 128)
```

---

### GCNEncoder

Graph Convolutional Network encoder.

```python
from src.models import GCNEncoder

gcn = GCNEncoder(
    input_dim=128,
    hidden_dim=128,
    output_dim=128
)

# Graph-aware encoding
z = gcn(features, edge_index)  # (batch_size, 128)
```

---

### ClassConditionedDecoder

Class-conditional decoder for reconstruction.

```python
from src.models import ClassConditionedDecoder

decoder = ClassConditionedDecoder(
    latent_dim=128,
    num_classes=5,
    output_length=187,
    class_embed_dim=32
)

# Reconstruct with class info
reconstructed = decoder(z, class_labels)  # (batch_size, 1, 187)
```

---

## Data Loading

### load_ecg_data

Load ECG data from CSV file.

```python
from src.data_loader import load_ecg_data

waveforms, labels = load_ecg_data(
    file_path='data/mitbih_train.csv',
    normalize=True  # Standardize features
)

# Returns:
#   waveforms: (N, 187) numpy array
#   labels: (N,) numpy array
```

---

### get_dataloaders

Create PyTorch DataLoaders.

```python
from src.data_loader import get_dataloaders

# With validation split
train_loader, val_loader = get_dataloaders(
    train_path='data/mitbih_train.csv',
    batch_size=512,
    val_split=0.15,
    normalize=True,
    num_workers=4
)

# With separate test set
train_loader, val_loader, test_loader = get_dataloaders(
    train_path='data/mitbih_train.csv',
    test_path='data/mitbih_test.csv',
    batch_size=512,
    val_split=0.15
)
```

---

### ECGDataset

PyTorch Dataset class.

```python
from src.data_loader import ECGDataset
from torch.utils.data import DataLoader

dataset = ECGDataset(waveforms, labels)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Iterate
for waveforms_batch, labels_batch in loader:
    # waveforms_batch: (32, 1, 187)
    # labels_batch: (32,)
    pass
```

---

### get_class_weights

Compute class weights for imbalanced data.

```python
from src.data_loader import get_class_weights

weights = get_class_weights(labels)  # Tensor of shape (num_classes,)

# Use in loss function
criterion = nn.CrossEntropyLoss(weight=weights)
```

---

## Graph Utilities

### build_knn_graph

Build k-nearest neighbor graph.

```python
from src.graph_utils import build_knn_graph

edge_index = build_knn_graph(
    embeddings,        # (batch_size, feature_dim)
    k=5,              # Number of neighbors
    metric='cosine'   # Distance metric
)

# Returns:
#   edge_index: (2, num_edges) - Graph edges in COO format
```

**Supported metrics:** `'cosine'`, `'euclidean'`, `'manhattan'`

---

### compute_edge_scores

Compute similarity scores for edges.

```python
from src.graph_utils import compute_edge_scores

scores = compute_edge_scores(
    embeddings,    # (N, feature_dim)
    edge_index     # (2, num_edges)
)

# Returns:
#   scores: (num_edges,) - Cosine similarity for each edge
```

---

### build_batch_knn_graph

Build graphs for batched data.

```python
from src.graph_utils import build_batch_knn_graph

# Batch indices indicate which batch each node belongs to
batch_indices = torch.tensor([0]*10 + [1]*10 + [2]*10)  # 3 batches of 10 nodes

edge_index = build_batch_knn_graph(
    embeddings,       # (30, 128)
    batch_indices,    # (30,)
    k=5
)
```

---

### visualize_graph

Visualize graph structure.

```python
from src.graph_utils import visualize_graph

fig = visualize_graph(
    embeddings,      # (N, feature_dim)
    edge_index,      # (2, num_edges)
    labels=labels,   # Optional: (N,) for coloring
    method='tsne'    # 'tsne' or 'umap'
)

fig.savefig('graph_visualization.png')
```

---

## Training

### train_model (Function)

Complete training pipeline.

```python
from src.train import train_model

train_model(
    train_path='data/mitbih_train.csv',
    val_split=0.15,
    epochs=200,
    batch_size=512,
    lr=0.001,
    k_neighbors=5,
    lambda_class=0.5,    # Classification loss weight
    lambda_graph=0.1,    # Graph loss weight
    device='cuda',
    save_path='models/best_model.pth'
)
```

### Command Line Training

```bash
python src/train.py \
    --train-data data/mitbih_train.csv \
    --epochs 200 \
    --batch-size 512 \
    --lr 0.001 \
    --k-neighbors 5 \
    --lambda-class 0.5 \
    --lambda-graph 0.1 \
    --device cuda \
    --save-path models/best_model.pth
```

---

## Inference

### ECGPredictor

High-level predictor class.

```python
from src.inference import ECGPredictor

# Initialize
predictor = ECGPredictor(
    model_path='models/best_model.pth',
    device='cuda'
)

# Predict single sample
prediction, confidence = predictor.predict(ecg_waveform)
print(f"Class: {prediction}, Confidence: {confidence:.2%}")

# Reconstruct
reconstructed = predictor.reconstruct(ecg_waveform)

# Extract latent representation
latent = predictor.encode(ecg_waveform)  # (128,)
```

---

### predict_from_csv

Batch prediction on CSV file.

```python
from src.inference import predict_from_csv

results = predict_from_csv(
    model_path='models/best_model.pth',
    input_csv='data/test.csv',
    output_csv='predictions.csv',
    batch_size=512,
    device='cuda'
)

# Results DataFrame contains:
#   - predicted_class
#   - confidence
#   - true_class (if available)
#   - correct (if true labels available)
```

### Command Line Inference

```bash
python src/inference.py \
    --model models/best_model.pth \
    --input data/test.csv \
    --output predictions.csv \
    --batch-size 512 \
    --device cuda
```

---

## Complete Example

### Training Script

```python
import torch
from src.models import EnhancedJointECGModel
from src.data_loader import get_dataloaders, get_class_weights, load_ecg_data
from src.graph_utils import build_knn_graph
from src.train import train_model

# Option 1: Use high-level function
train_model(
    train_path='data/mitbih_train.csv',
    epochs=200,
    batch_size=512,
    device='cuda'
)

# Option 2: Manual training loop
device = torch.device('cuda')

# Load data
train_loader, val_loader = get_dataloaders('data/mitbih_train.csv')

# Initialize model
model = EnhancedJointECGModel().to(device)

# Training loop
for epoch in range(200):
    for waveforms, labels in train_loader:
        waveforms, labels = waveforms.to(device), labels.to(device)
        
        # Build graph
        with torch.no_grad():
            features = model.ecg_encoder(waveforms)
            edge_index = build_knn_graph(features, k=5).to(device)
        
        # Forward pass
        z, recon, logits = model(waveforms, edge_index, labels)
        
        # Compute loss and backprop
        # ... (see src/train.py for full implementation)
```

### Inference Script

```python
from src.inference import ECGPredictor
import numpy as np

# Load predictor
predictor = ECGPredictor('models/best_model.pth')

# Single prediction
ecg = np.random.randn(187)  # Your ECG data
prediction, confidence = predictor.predict(ecg)

print(f"Predicted: {prediction}")
print(f"Confidence: {confidence:.2%}")

# Batch prediction from file
from src.inference import predict_from_csv

results = predict_from_csv(
    model_path='models/best_model.pth',
    input_csv='data/test.csv',
    output_csv='predictions.csv'
)

print(f"Accuracy: {results['correct'].mean():.2%}")
```

---

## Utilities

### Save/Load Model

```python
# Save
torch.save(model.state_dict(), 'model.pth')

# Load
model = EnhancedJointECGModel()
model.load_state_dict(torch.load('model.pth'))
model.eval()
```

### Export to ONNX

```python
import torch.onnx

model.eval()
dummy_input = torch.randn(1, 1, 187)
dummy_edge_index = torch.LongTensor([[0], [0]])

torch.onnx.export(
    model,
    (dummy_input, dummy_edge_index, torch.LongTensor([0])),
    'model.onnx',
    input_names=['waveform', 'edge_index', 'class_label'],
    output_names=['latent', 'reconstructed', 'logits'],
    dynamic_axes={'waveform': {0: 'batch_size'}}
)
```

---

## Error Handling

```python
try:
    prediction, confidence = predictor.predict(ecg_data)
except ValueError as e:
    print(f"Invalid input shape: {e}")
except FileNotFoundError as e:
    print(f"Model not found: {e}")
except RuntimeError as e:
    print(f"CUDA error: {e}")
```

---

## Tips & Best Practices

### Memory Optimization

```python
# Use mixed precision training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    z, recon, logits = model(waveforms, edge_index, labels)
    loss = criterion(recon, waveforms)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Batch Size Selection

```python
# GPU memory considerations
if torch.cuda.is_available():
    gpu_memory = torch.cuda.get_device_properties(0).total_memory
    if gpu_memory < 8e9:  # < 8GB
        batch_size = 256
    else:
        batch_size = 512
else:
    batch_size = 128  # CPU
```

### Data Augmentation

```python
def augment_ecg(waveform):
    """Simple ECG augmentation."""
    # Add noise
    noise = torch.randn_like(waveform) * 0.01
    waveform = waveform + noise
    
    # Scale
    scale = 1.0 + (torch.rand(1) - 0.5) * 0.2
    waveform = waveform * scale
    
    return waveform
```

---

[Back to Main README](../README.md) | [Methodology](METHODOLOGY.md) | [Results](RESULTS.md)
