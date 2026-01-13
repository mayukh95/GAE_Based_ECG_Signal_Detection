"""
Model Architectures for ECG Classification

This module contains all neural network architectures:
- ECGEncoder: CNN-based feature extractor
- GCNEncoder: Graph Convolutional Network encoder
- ClassConditionedDecoder: Decoder that uses class information
- SimpleClassifier: Classification head
- EnhancedJointECGModel: Complete pipeline
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class ECGEncoder(nn.Module):
    """
    CNN-based encoder for ECG waveform feature extraction.
    
    Applies three 1D convolutional layers with batch normalization,
    ReLU activation, max pooling, and dropout for robust feature learning.
    
    Args:
        input_length (int): Length of input ECG waveform (default: 187)
        latent_dim (int): Output feature dimension (default: 128)
        dropout_rate (float): Dropout probability (default: 0.3)
    
    Input Shape:
        (batch_size, 1, input_length)
    
    Output Shape:
        (batch_size, latent_dim)
    
    Example:
        >>> encoder = ECGEncoder(input_length=187, latent_dim=128)
        >>> x = torch.randn(32, 1, 187)  # batch of 32 ECG waveforms
        >>> features = encoder(x)  # (32, 128)
    """
    
    def __init__(self, input_length=187, latent_dim=128, dropout_rate=0.3):
        super(ECGEncoder, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)
        
        self.conv3 = nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(2)
        
        # Calculate flattened dimension after convolutions
        self.flatten_dim = 128 * (input_length // 8)
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flatten_dim, 256)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, latent_dim)
        
    def forward(self, x):
        """Forward pass through the encoder."""
        # Conv block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # Conv block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        # Conv block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)
        
        # Flatten and fully connected
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        
        return x


class GCNEncoder(nn.Module):
    """
    Graph Convolutional Network encoder.
    
    Applies graph convolutions to aggregate information from neighboring
    nodes in the k-NN graph, creating graph-aware representations.
    
    Args:
        input_dim (int): Input feature dimension
        hidden_dim (int): Hidden layer dimension (default: 128)
        output_dim (int): Output feature dimension (default: 128)
    
    Example:
        >>> gcn = GCNEncoder(input_dim=128, output_dim=128)
        >>> x = torch.randn(100, 128)  # 100 nodes, 128 features each
        >>> edge_index = torch.randint(0, 100, (2, 500))  # Graph edges
        >>> z = gcn(x, edge_index)  # (100, 128) graph-aware features
    """
    
    def __init__(self, input_dim, hidden_dim=128, output_dim=128):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        
    def forward(self, x, edge_index):
        """
        Forward pass through GCN layers.
        
        Args:
            x: Node features (batch_size, input_dim)
            edge_index: Graph connectivity (2, num_edges)
            
        Returns:
            Graph-aware node embeddings (batch_size, output_dim)
        """
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x


class ClassConditionedDecoder(nn.Module):
    """
    Decoder that uses class information for reconstruction.
    
    This is the KEY INNOVATION: unlike standard autoencoders, this decoder
    takes class labels as input and learns class-specific reconstruction
    patterns. This allows the model to generate more accurate reconstructions
    tailored to each arrhythmia type.
    
    Args:
        latent_dim (int): Dimension of latent code (default: 128)
        num_classes (int): Number of arrhythmia classes (default: 5)
        output_length (int): Length of output waveform (default: 187)
        class_embed_dim (int): Dimension of class embeddings (default: 32)
    
    Example:
        >>> decoder = ClassConditionedDecoder(latent_dim=128, num_classes=5)
        >>> z = torch.randn(32, 128)  # Latent codes
        >>> labels = torch.randint(0, 5, (32,))  # Class labels
        >>> reconstructed = decoder(z, labels)  # (32, 1, 187)
    """
    
    def __init__(self, latent_dim=128, num_classes=5, output_length=187, 
                 class_embed_dim=32):
        super(ClassConditionedDecoder, self).__init__()
        
        # Class embedding layer
        self.class_embedding = nn.Embedding(num_classes, class_embed_dim)
        
        # Combined input: latent code + class embedding
        combined_dim = latent_dim + class_embed_dim
        
        # Fully connected layers
        self.fc1 = nn.Linear(combined_dim, 256)
        self.fc2 = nn.Linear(256, 128 * (output_length // 8))
        
        self.output_length = output_length
        
        # Transposed convolutions (deconvolutions)
        self.deconv1 = nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        
        self.deconv2 = nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        
        self.deconv3 = nn.ConvTranspose1d(32, 1, kernel_size=4, stride=2, padding=1)
        
    def forward(self, z, class_labels):
        """
        Forward pass with class conditioning.
        
        Args:
            z: Latent code (batch_size, latent_dim)
            class_labels: Class labels (batch_size,)
            
        Returns:
            Reconstructed waveform (batch_size, 1, output_length)
        """
        # Get class embeddings
        class_emb = self.class_embedding(class_labels)
        
        # Concatenate latent code with class embedding
        x = torch.cat([z, class_emb], dim=1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # Reshape for deconvolution
        x = x.view(x.size(0), 128, -1)
        
        # Deconvolutional layers
        x = self.deconv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        x = self.deconv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        x = self.deconv3(x)
        
        # Trim or pad to exact output length
        if x.size(2) > self.output_length:
            x = x[:, :, :self.output_length]
        elif x.size(2) < self.output_length:
            padding = self.output_length - x.size(2)
            x = F.pad(x, (0, padding))
        
        return x


class SimpleClassifier(nn.Module):
    """
    Multi-layer perceptron for classification.
    
    Takes latent representations and predicts arrhythmia class.
    
    Args:
        latent_dim (int): Input dimension
        num_classes (int): Number of output classes
    """
    
    def __init__(self, latent_dim=128, num_classes=5):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 64)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, z):
        """Classify latent representations."""
        x = F.relu(self.fc1(z))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class EnhancedJointECGModel(nn.Module):
    """
    Complete Class-Conditional Graph Autoencoder for ECG Classification.
    
    This is the main model that combines all components:
    1. CNN encoder for feature extraction
    2. GCN encoder for graph-aware representations
    3. Classifier for arrhythmia prediction
    4. Class-conditioned decoder for reconstruction
    
    Args:
        input_length (int): ECG waveform length (default: 187)
        latent_dim (int): Latent space dimension (default: 128)
        num_classes (int): Number of arrhythmia classes (default: 5)
        dropout_rate (float): Dropout probability (default: 0.3)
    
    Training Mode:
        Uses TRUE labels for class-conditioned reconstruction
        
    Inference Mode:
        Uses PREDICTED labels for class-conditioned reconstruction
    
    Example:
        >>> model = EnhancedJointECGModel()
        >>> 
        >>> # Training
        >>> x = torch.randn(32, 1, 187)
        >>> edge_index = build_knn_graph(x, k=5)
        >>> true_labels = torch.randint(0, 5, (32,))
        >>> z, recon, logits = model(x, edge_index, true_labels)
        >>> 
        >>> # Inference
        >>> z, recon, logits, pred_labels = model.forward_inference(x, edge_index)
    """
    
    def __init__(self, input_length=187, latent_dim=128, num_classes=5, 
                 dropout_rate=0.3):
        super(EnhancedJointECGModel, self).__init__()
        
        self.input_length = input_length
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        # Initialize all components
        self.ecg_encoder = ECGEncoder(input_length, latent_dim, dropout_rate)
        self.gcn_encoder = GCNEncoder(latent_dim, latent_dim, latent_dim)
        self.classifier = SimpleClassifier(latent_dim, num_classes)
        self.decoder = ClassConditionedDecoder(latent_dim, num_classes, input_length)
        
    def forward(self, waveforms, edge_index, class_labels):
        """
        Forward pass for TRAINING (uses true labels).
        
        Args:
            waveforms: Input ECG (batch_size, 1, input_length)
            edge_index: Graph edges (2, num_edges)
            class_labels: True class labels (batch_size,)
            
        Returns:
            z: Latent representations
            reconstructed: Reconstructed waveforms
            logits: Classification logits
        """
        # Encode with CNN
        cnn_features = self.ecg_encoder(waveforms)
        
        # Encode with GCN (graph-aware)
        z = self.gcn_encoder(cnn_features, edge_index)
        
        # Classify
        logits = self.classifier(z)
        
        # Reconstruct using TRUE labels (teacher forcing during training)
        reconstructed = self.decoder(z, class_labels)
        
        return z, reconstructed, logits
    
    def forward_inference(self, waveforms, edge_index):
        """
        Forward pass for INFERENCE (uses predicted labels).
        
        This is used during evaluation and deployment when true labels
        are not available.
        
        Args:
            waveforms: Input ECG (batch_size, 1, input_length)
            edge_index: Graph edges (2, num_edges)
            
        Returns:
            z: Latent representations
            reconstructed: Reconstructed waveforms
            logits: Classification logits
            predicted_classes: Predicted class labels
        """
        with torch.no_grad():
            # Encode
            cnn_features = self.ecg_encoder(waveforms)
            z = self.gcn_encoder(cnn_features, edge_index)
            
            # Classify
            logits = self.classifier(z)
            predicted_classes = torch.argmax(logits, dim=1)
            
            # Reconstruct using PREDICTED labels
            reconstructed = self.decoder(z, predicted_classes)
        
        return z, reconstructed, logits, predicted_classes
    
    def encode(self, waveforms, edge_index):
        """Extract latent representations only."""
        cnn_features = self.ecg_encoder(waveforms)
        z = self.gcn_encoder(cnn_features, edge_index)
        return z
    
    def predict(self, waveforms, edge_index):
        """Get predictions only."""
        z = self.encode(waveforms, edge_index)
        logits = self.classifier(z)
        return torch.softmax(logits, dim=1)
