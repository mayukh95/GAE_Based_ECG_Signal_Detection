"""
Inference and Prediction Script

Usage:
    python src/inference.py --model models/best_model.pth --input test.csv --output predictions.csv
"""

import argparse
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

from src.models import EnhancedJointECGModel
from src.data_loader import ECGDataset
from src.graph_utils import build_knn_graph
from torch.utils.data import DataLoader


class ECGPredictor:
    """
    Easy-to-use predictor class for ECG classification.
    
    Example:
        >>> predictor = ECGPredictor('models/best_model.pth')
        >>> prediction, confidence = predictor.predict(ecg_waveform)
        >>> print(f"Predicted: {prediction} (confidence: {confidence:.2%})")
    """
    
    def __init__(self, model_path, device='cuda'):
        """
        Initialize predictor.
        
        Args:
            model_path: Path to trained model weights
            device: Device to use ('cuda' or 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = EnhancedJointECGModel(
            input_length=187,
            latent_dim=128,
            num_classes=5
        ).to(self.device)
        
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        self.class_names = ['Normal', 'Supraventricular', 'Ventricular', 'Fusion', 'Unknown']
    
    def predict(self, waveform, k_neighbors=5):
        """
        Predict class for a single ECG waveform.
        
        Args:
            waveform: ECG waveform (187,) or (1, 187) numpy array or torch tensor
            k_neighbors: Number of neighbors for graph (not used for single sample)
            
        Returns:
            predicted_class: Predicted class name
            confidence: Prediction confidence (0-1)
        """
        # Preprocess input
        if isinstance(waveform, np.ndarray):
            waveform = torch.FloatTensor(waveform)
        
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0).unsqueeze(0)  # (1, 1, 187)
        elif waveform.ndim == 2:
            waveform = waveform.unsqueeze(1)  # (N, 1, 187)
        
        waveform = waveform.to(self.device)
        
        # For single sample, create a simple graph (self-loop)
        if waveform.size(0) == 1:
            edge_index = torch.LongTensor([[0], [0]]).to(self.device)
        else:
            # For batch, build k-NN graph
            with torch.no_grad():
                cnn_features = self.model.ecg_encoder(waveform)
                edge_index = build_knn_graph(cnn_features, k=k_neighbors)
                edge_index = edge_index.to(self.device)
        
        # Predict
        with torch.no_grad():
            _, _, logits, predicted = self.model.forward_inference(waveform, edge_index)
            probabilities = torch.softmax(logits, dim=1)
            confidence = probabilities.max(dim=1)[0]
        
        # Get class name
        if waveform.size(0) == 1:
            predicted_class = self.class_names[predicted.item()]
            confidence = confidence.item()
        else:
            predicted_class = [self.class_names[p.item()] for p in predicted]
            confidence = confidence.cpu().numpy()
        
        return predicted_class, confidence
    
    def reconstruct(self, waveform, k_neighbors=5):
        """
        Reconstruct ECG waveform using the autoencoder.
        
        Args:
            waveform: ECG waveform
            
        Returns:
            reconstructed: Reconstructed waveform
        """
        # Preprocess
        if isinstance(waveform, np.ndarray):
            waveform = torch.FloatTensor(waveform)
        
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0).unsqueeze(0)
        elif waveform.ndim == 2:
            waveform = waveform.unsqueeze(1)
        
        waveform = waveform.to(self.device)
        
        # Build graph
        if waveform.size(0) == 1:
            edge_index = torch.LongTensor([[0], [0]]).to(self.device)
        else:
            with torch.no_grad():
                cnn_features = self.model.ecg_encoder(waveform)
                edge_index = build_knn_graph(cnn_features, k=k_neighbors)
                edge_index = edge_index.to(self.device)
        
        # Reconstruct
        with torch.no_grad():
            _, reconstructed, _, _ = self.model.forward_inference(waveform, edge_index)
        
        return reconstructed.squeeze().cpu().numpy()
    
    def encode(self, waveform, k_neighbors=5):
        """
        Extract latent representation.
        
        Args:
            waveform: ECG waveform
            
        Returns:
            latent: Latent representation (128D)
        """
        # Preprocess
        if isinstance(waveform, np.ndarray):
            waveform = torch.FloatTensor(waveform)
        
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0).unsqueeze(0)
        elif waveform.ndim == 2:
            waveform = waveform.unsqueeze(1)
        
        waveform = waveform.to(self.device)
        
        # Build graph
        if waveform.size(0) == 1:
            edge_index = torch.LongTensor([[0], [0]]).to(self.device)
        else:
            with torch.no_grad():
                cnn_features = self.model.ecg_encoder(waveform)
                edge_index = build_knn_graph(cnn_features, k=k_neighbors)
                edge_index = edge_index.to(self.device)
        
        # Encode
        with torch.no_grad():
            latent = self.model.encode(waveform, edge_index)
        
        return latent.cpu().numpy()


def predict_from_csv(model_path, input_csv, output_csv=None, batch_size=512, device='cuda'):
    """
    Predict on entire CSV file.
    
    Args:
        model_path: Path to trained model
        input_csv: Path to input CSV (same format as training data)
        output_csv: Path to save predictions (optional)
        batch_size: Batch size for inference
        device: Device to use
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Load model
    print("Loading model...")
    model = EnhancedJointECGModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Load data
    print("Loading data...")
    df = pd.read_csv(input_csv, header=None)
    waveforms = df.iloc[:, :-1].values
    true_labels = df.iloc[:, -1].values.astype(int) if df.shape[1] > 187 else None
    
    # Create dataset
    if true_labels is not None:
        dataset = ECGDataset(waveforms, true_labels)
    else:
        dataset = ECGDataset(waveforms, np.zeros(len(waveforms)))
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Predict
    print("Running inference...")
    all_predictions = []
    all_confidences = []
    
    with torch.no_grad():
        for batch_waveforms, _ in tqdm(dataloader):
            batch_waveforms = batch_waveforms.to(device)
            
            # Build graph
            cnn_features = model.ecg_encoder(batch_waveforms)
            edge_index = build_knn_graph(cnn_features, k=5)
            edge_index = edge_index.to(device)
            
            # Predict
            _, _, logits, predicted = model.forward_inference(batch_waveforms, edge_index)
            probabilities = torch.softmax(logits, dim=1)
            confidences = probabilities.max(dim=1)[0]
            
            all_predictions.extend(predicted.cpu().numpy())
            all_confidences.extend(confidences.cpu().numpy())
    
    # Create results dataframe
    results = pd.DataFrame({
        'predicted_class': all_predictions,
        'confidence': all_confidences
    })
    
    if true_labels is not None:
        results['true_class'] = true_labels
        results['correct'] = results['predicted_class'] == results['true_class']
        accuracy = results['correct'].mean()
        print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Save results
    if output_csv:
        results.to_csv(output_csv, index=False)
        print(f"Results saved to: {output_csv}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='ECG Classification Inference')
    
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to input CSV file')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save predictions')
    parser.add_argument('--batch-size', type=int, default=512,
                       help='Batch size for inference')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use')
    
    args = parser.parse_args()
    
    # Run inference
    predict_from_csv(
        model_path=args.model,
        input_csv=args.input,
        output_csv=args.output,
        batch_size=args.batch_size,
        device=args.device
    )


if __name__ == '__main__':
    main()
