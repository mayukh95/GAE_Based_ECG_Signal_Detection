"""
Script to download MIT-BIH Arrhythmia Database from PhysioNet

Usage:
    python scripts/download_data.py --output data/
"""

import argparse
import os
import urllib.request
import pandas as pd
import numpy as np


def download_file(url, output_path):
    """Download a file from URL."""
    print(f"Downloading {url}...")
    urllib.request.urlretrieve(url, output_path)
    print(f"Saved to {output_path}")


def download_mitbih_dataset(output_dir='data/'):
    """
    Download MIT-BIH Arrhythmia Database from Kaggle/PhysioNet.
    
    Note: The dataset is typically obtained from:
    - Kaggle: https://www.kaggle.com/shayanfazeli/heartbeat
    - PhysioNet: https://physionet.org/content/mitdb/1.0.0/
    
    Args:
        output_dir: Directory to save downloaded files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("MIT-BIH Arrhythmia Database Download")
    print("=" * 60)
    
    print("\nIMPORTANT:")
    print("Due to licensing and hosting considerations, please download")
    print("the dataset manually from one of these sources:\n")
    
    print("Option 1 - Kaggle (Preprocessed):")
    print("  1. Visit: https://www.kaggle.com/shayanfazeli/heartbeat")
    print("  2. Download: mitbih_train.csv and mitbih_test.csv")
    print(f"  3. Place files in: {os.path.abspath(output_dir)}\n")
    
    print("Option 2 - PhysioNet (Raw):")
    print("  1. Visit: https://physionet.org/content/mitdb/1.0.0/")
    print("  2. Download raw ECG records")
    print("  3. Use WFDB library to process signals\n")
    
    print("=" * 60)
    
    # Check if files already exist
    train_file = os.path.join(output_dir, 'mitbih_train.csv')
    test_file = os.path.join(output_dir, 'mitbih_test.csv')
    
    if os.path.exists(train_file) and os.path.exists(test_file):
        print("\n✓ Dataset files found!")
        print(f"  Train: {train_file}")
        print(f"  Test: {test_file}")
        
        # Verify data
        train_df = pd.read_csv(train_file, header=None)
        test_df = pd.read_csv(test_file, header=None)
        
        print(f"\nDataset Statistics:")
        print(f"  Training samples: {len(train_df):,}")
        print(f"  Test samples: {len(test_df):,}")
        print(f"  Feature dimensions: {train_df.shape[1] - 1}")
        print(f"  Number of classes: {train_df.iloc[:, -1].nunique()}")
        
    else:
        print("\n⚠ Dataset files not found.")
        print("Please follow the instructions above to download the dataset.")


def create_sample_data(output_dir='examples/'):
    """
    Create a small sample dataset for testing.
    
    This generates synthetic ECG-like data for demonstration purposes.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nGenerating sample data for testing...")
    
    # Generate synthetic ECG waveforms
    n_samples = 100
    n_features = 187
    
    # Simple synthetic ECG patterns
    samples = []
    labels = []
    
    for i in range(n_samples):
        # Random class
        label = np.random.randint(0, 5)
        
        # Generate synthetic ECG
        t = np.linspace(0, 1, n_features)
        
        # QRS complex (simplified)
        qrs = np.exp(-((t - 0.3) ** 2) / 0.01) * (1 if label == 0 else 1.5)
        
        # T wave
        t_wave = np.exp(-((t - 0.6) ** 2) / 0.02) * 0.5
        
        # Add class-specific variations
        if label == 1:  # Supraventricular
            qrs *= 0.8
        elif label == 2:  # Ventricular
            qrs *= 1.5
            qrs = np.roll(qrs, 10)
        elif label == 3:  # Fusion
            qrs *= 1.2
        
        # Combine and add noise
        ecg = qrs + t_wave + np.random.randn(n_features) * 0.05
        
        samples.append(ecg)
        labels.append(label)
    
    # Create DataFrame
    df = pd.DataFrame(samples)
    df['label'] = labels
    
    # Save
    output_file = os.path.join(output_dir, 'sample_ecgs.csv')
    df.to_csv(output_file, index=False, header=False)
    
    print(f"✓ Sample data created: {output_file}")
    print(f"  Samples: {n_samples}")
    print(f"  Features: {n_features}")
    print(f"  Classes: {df['label'].nunique()}")


def main():
    parser = argparse.ArgumentParser(
        description='Download MIT-BIH Arrhythmia Database'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='data/',
        help='Output directory for dataset (default: data/)'
    )
    
    parser.add_argument(
        '--create-sample',
        action='store_true',
        help='Create sample synthetic data for testing'
    )
    
    args = parser.parse_args()
    
    # Download main dataset
    download_mitbih_dataset(args.output)
    
    # Optionally create sample data
    if args.create_sample:
        create_sample_data('examples/')


if __name__ == '__main__':
    main()
