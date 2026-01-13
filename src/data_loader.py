"""
Data Loading and Processing Utilities

This module provides functions for loading ECG data and creating PyTorch dataloaders.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class ECGDataset(Dataset):
    """
    PyTorch Dataset for ECG waveforms.
    
    Args:
        waveforms: ECG waveforms (N, seq_length)
        labels: Class labels (N,)
        transform: Optional transform to apply
    
    Example:
        >>> dataset = ECGDataset(waveforms, labels)
        >>> dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    """
    
    def __init__(self, waveforms, labels, transform=None):
        self.waveforms = torch.FloatTensor(waveforms).unsqueeze(1)  # Add channel dim
        self.labels = torch.LongTensor(labels)
        self.transform = transform
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        waveform = self.waveforms[idx]
        label = self.labels[idx]
        
        if self.transform:
            waveform = self.transform(waveform)
        
        return waveform, label


def load_ecg_data(file_path, normalize=True):
    """
    Load ECG data from CSV file.
    
    Expected format: Each row is a sample, last column is the label,
    all other columns are the ECG waveform features.
    
    Args:
        file_path (str): Path to CSV file
        normalize (bool): Whether to standardize features
        
    Returns:
        waveforms: ECG waveforms array (N, seq_length)
        labels: Class labels array (N,)
    
    Example:
        >>> waveforms, labels = load_ecg_data('data/mitbih_train.csv')
    """
    # Load data
    df = pd.read_csv(file_path, header=None)
    
    # Separate features and labels
    waveforms = df.iloc[:, :-1].values
    labels = df.iloc[:, -1].values.astype(int)
    
    # Normalize waveforms
    if normalize:
        scaler = StandardScaler()
        waveforms = scaler.fit_transform(waveforms)
    
    return waveforms, labels


def get_dataloaders(train_path, test_path=None, batch_size=512, 
                    val_split=0.15, normalize=True, num_workers=4):
    """
    Create train/val/test dataloaders from CSV files.
    
    Args:
        train_path (str): Path to training data CSV
        test_path (str, optional): Path to test data CSV
        batch_size (int): Batch size for dataloaders
        val_split (float): Fraction of train data to use for validation
        normalize (bool): Whether to standardize features
        num_workers (int): Number of dataloader workers
        
    Returns:
        If test_path is provided:
            train_loader, val_loader, test_loader
        Otherwise:
            train_loader, val_loader
    
    Example:
        >>> train_loader, val_loader = get_dataloaders(
        ...     train_path='data/train.csv',
        ...     batch_size=512
        ... )
    """
    # Load training data
    waveforms, labels = load_ecg_data(train_path, normalize=normalize)
    
    # Split into train and validation
    if val_split > 0:
        train_waveforms, val_waveforms, train_labels, val_labels = train_test_split(
            waveforms, labels, test_size=val_split, random_state=42, stratify=labels
        )
    else:
        train_waveforms, train_labels = waveforms, labels
        val_waveforms, val_labels = None, None
    
    # Create datasets
    train_dataset = ECGDataset(train_waveforms, train_labels)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    if val_waveforms is not None:
        val_dataset = ECGDataset(val_waveforms, val_labels)
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=True
        )
    else:
        val_loader = None
    
    # Load test data if provided
    if test_path is not None:
        test_waveforms, test_labels = load_ecg_data(test_path, normalize=normalize)
        test_dataset = ECGDataset(test_waveforms, test_labels)
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=True
        )
        return train_loader, val_loader, test_loader
    
    return train_loader, val_loader


def get_class_weights(labels):
    """
    Compute class weights for handling imbalanced data.
    
    Args:
        labels: Array of class labels
        
    Returns:
        Tensor of class weights
    
    Example:
        >>> labels = np.array([0, 0, 0, 1, 1, 2])
        >>> weights = get_class_weights(labels)
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = np.unique(labels)
    weights = compute_class_weight('balanced', classes=classes, y=labels)
    return torch.FloatTensor(weights)
