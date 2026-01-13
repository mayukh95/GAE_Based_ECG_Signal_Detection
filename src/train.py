"""
Training Script for ECG Classification Model

Usage:
    python src/train.py --train-data data/mitbih_train.csv --epochs 200 --batch-size 512
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np

from src.models import EnhancedJointECGModel
from src.data_loader import get_dataloaders, get_class_weights, load_ecg_data
from src.graph_utils import build_knn_graph, compute_edge_scores


def train_epoch(model, train_loader, optimizer, criterion_recon, criterion_class,
                device, k_neighbors=5, lambda_class=0.5, lambda_graph=0.1):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_recon_loss = 0
    total_class_loss = 0
    total_graph_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    
    for batch_idx, (waveforms, labels) in enumerate(pbar):
        waveforms = waveforms.to(device)
        labels = labels.to(device)
        
        # Extract CNN features and build graph
        with torch.no_grad():
            cnn_features = model.ecg_encoder(waveforms)
            edge_index = build_knn_graph(cnn_features, k=k_neighbors)
            edge_index = edge_index.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        z, reconstructed, logits = model(waveforms, edge_index, labels)
        
        # Compute losses
        # 1. Reconstruction loss
        loss_recon = criterion_recon(reconstructed, waveforms)
        
        # 2. Classification loss
        loss_class = criterion_class(logits, labels)
        
        # 3. Graph reconstruction loss (preserve similarity structure)
        edge_scores = compute_edge_scores(z, edge_index)
        loss_graph = -torch.mean(edge_scores)  # Maximize edge similarity
        
        # Total loss
        loss = loss_recon + lambda_class * loss_class + lambda_graph * loss_graph
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        total_recon_loss += loss_recon.item()
        total_class_loss += loss_class.item()
        total_graph_loss += loss_graph.item()
        
        _, predicted = torch.max(logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    avg_loss = total_loss / len(train_loader)
    avg_recon_loss = total_recon_loss / len(train_loader)
    avg_class_loss = total_class_loss / len(train_loader)
    avg_graph_loss = total_graph_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, avg_recon_loss, avg_class_loss, avg_graph_loss, accuracy


def validate(model, val_loader, criterion_recon, criterion_class, 
             device, k_neighbors=5, lambda_class=0.5, lambda_graph=0.1):
    """Validate the model."""
    model.eval()
    total_loss = 0
    total_recon_loss = 0
    total_class_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for waveforms, labels in tqdm(val_loader, desc='Validation'):
            waveforms = waveforms.to(device)
            labels = labels.to(device)
            
            # Extract features and build graph
            cnn_features = model.ecg_encoder(waveforms)
            edge_index = build_knn_graph(cnn_features, k=k_neighbors)
            edge_index = edge_index.to(device)
            
            # Forward pass
            z, reconstructed, logits = model(waveforms, edge_index, labels)
            
            # Compute losses
            loss_recon = criterion_recon(reconstructed, waveforms)
            loss_class = criterion_class(logits, labels)
            edge_scores = compute_edge_scores(z, edge_index)
            loss_graph = -torch.mean(edge_scores)
            
            loss = loss_recon + lambda_class * loss_class + lambda_graph * loss_graph
            
            total_loss += loss.item()
            total_recon_loss += loss_recon.item()
            total_class_loss += loss_class.item()
            
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(val_loader)
    avg_recon_loss = total_recon_loss / len(val_loader)
    avg_class_loss = total_class_loss / len(val_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, avg_recon_loss, avg_class_loss, accuracy


def train_model(train_path, val_split=0.15, epochs=200, batch_size=512,
                lr=0.001, k_neighbors=5, lambda_class=0.5, lambda_graph=0.1,
                device='cuda', save_path='models/best_model.pth'):
    """
    Complete training pipeline.
    
    Args:
        train_path: Path to training data CSV
        val_split: Validation split ratio
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        k_neighbors: Number of neighbors in k-NN graph
        lambda_class: Weight for classification loss
        lambda_graph: Weight for graph loss
        device: Device to use ('cuda' or 'cpu')
        save_path: Path to save best model
    """
    # Set device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    train_loader, val_loader = get_dataloaders(
        train_path=train_path,
        batch_size=batch_size,
        val_split=val_split
    )
    
    # Get class weights for imbalanced data
    _, labels = load_ecg_data(train_path)
    class_weights = get_class_weights(labels).to(device)
    
    # Initialize model
    print("Initializing model...")
    model = EnhancedJointECGModel(
        input_length=187,
        latent_dim=128,
        num_classes=5
    ).to(device)
    
    # Loss functions
    criterion_recon = nn.MSELoss()
    criterion_class = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10, verbose=True
    )
    
    # Training loop
    print(f"\nTraining for {epochs} epochs...")
    best_val_acc = 0
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Train
        train_loss, train_recon, train_class, train_graph, train_acc = train_epoch(
            model, train_loader, optimizer, criterion_recon, criterion_class,
            device, k_neighbors, lambda_class, lambda_graph
        )
        
        # Validate
        val_loss, val_recon, val_class, val_acc = validate(
            model, val_loader, criterion_recon, criterion_class,
            device, k_neighbors, lambda_class, lambda_graph
        )
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  ├─ Recon: {train_recon:.4f} | Class: {train_class:.4f} | Graph: {train_graph:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"  ├─ Recon: {val_recon:.4f} | Class: {val_class:.4f}")
        
        # Learning rate scheduling
        scheduler.step(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"✓ Saved best model (Val Acc: {val_acc:.2f}%)")
    
    print(f"\nTraining complete! Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Train ECG Classification Model')
    
    # Data arguments
    parser.add_argument('--train-data', type=str, required=True,
                       help='Path to training data CSV')
    parser.add_argument('--val-split', type=float, default=0.15,
                       help='Validation split ratio (default: 0.15)')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=200,
                       help='Number of epochs (default: 200)')
    parser.add_argument('--batch-size', type=int, default=512,
                       help='Batch size (default: 512)')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    
    # Model arguments
    parser.add_argument('--k-neighbors', type=int, default=5,
                       help='Number of neighbors in k-NN graph (default: 5)')
    parser.add_argument('--lambda-class', type=float, default=0.5,
                       help='Weight for classification loss (default: 0.5)')
    parser.add_argument('--lambda-graph', type=float, default=0.1,
                       help='Weight for graph loss (default: 0.1)')
    
    # Device arguments
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use (default: cuda)')
    parser.add_argument('--save-path', type=str, default='models/best_model.pth',
                       help='Path to save model (default: models/best_model.pth)')
    
    args = parser.parse_args()
    
    # Train model
    train_model(
        train_path=args.train_data,
        val_split=args.val_split,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        k_neighbors=args.k_neighbors,
        lambda_class=args.lambda_class,
        lambda_graph=args.lambda_graph,
        device=args.device,
        save_path=args.save_path
    )


if __name__ == '__main__':
    main()
