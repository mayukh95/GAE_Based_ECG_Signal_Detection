"""
ECG Arrhythmia Classification Package

This package provides deep learning models for ECG arrhythmia classification
using class-conditional graph autoencoders.
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from src.models import (
    ECGEncoder,
    GCNEncoder,
    ClassConditionedDecoder,
    SimpleClassifier,
    EnhancedJointECGModel
)

from src.data_loader import (
    get_dataloaders,
    ECGDataset
)

from src.graph_utils import (
    build_knn_graph,
    compute_edge_scores
)

__all__ = [
    'ECGEncoder',
    'GCNEncoder',
    'ClassConditionedDecoder',
    'SimpleClassifier',
    'EnhancedJointECGModel',
    'get_dataloaders',
    'ECGDataset',
    'build_knn_graph',
    'compute_edge_scores'
]
