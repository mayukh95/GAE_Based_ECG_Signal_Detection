"""
Graph Construction Utilities

Functions for building k-NN graphs from ECG embeddings.
"""

import torch
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
import numpy as np


def build_knn_graph(embeddings, k=5, metric='cosine'):
    """
    Build a k-nearest neighbors graph from embeddings.
    
    Creates a graph where each node is connected to its k most similar
    neighbors based on the specified similarity metric.
    
    Args:
        embeddings: Node features (batch_size, feature_dim) or (N, feature_dim)
        k: Number of nearest neighbors
        metric: Distance metric ('cosine', 'euclidean', 'manhattan')
        
    Returns:
        edge_index: Graph edges in COO format (2, num_edges)
        
    Example:
        >>> embeddings = torch.randn(100, 128)
        >>> edge_index = build_knn_graph(embeddings, k=5)
        >>> print(edge_index.shape)  # (2, ~500)
    """
    # Convert to numpy if tensor
    if isinstance(embeddings, torch.Tensor):
        embeddings_np = embeddings.detach().cpu().numpy()
    else:
        embeddings_np = embeddings
    
    # Build k-NN index
    if metric == 'cosine':
        # Normalize for cosine similarity
        embeddings_norm = embeddings_np / (np.linalg.norm(embeddings_np, axis=1, keepdims=True) + 1e-8)
        nbrs = NearestNeighbors(n_neighbors=k+1, metric='cosine', algorithm='brute')
        nbrs.fit(embeddings_norm)
        _, indices = nbrs.kneighbors(embeddings_norm)
    else:
        nbrs = NearestNeighbors(n_neighbors=k+1, metric=metric)
        nbrs.fit(embeddings_np)
        _, indices = nbrs.kneighbors(embeddings_np)
    
    # Build edge list (exclude self-loops by skipping first neighbor)
    edge_list = []
    for i in range(len(embeddings_np)):
        for j in indices[i][1:]:  # Skip first neighbor (self)
            edge_list.append([i, j])
    
    # Convert to tensor
    edge_index = torch.LongTensor(edge_list).t().contiguous()
    
    # Make graph undirected (add reverse edges)
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    
    # Remove duplicate edges
    edge_index = torch.unique(edge_index, dim=1)
    
    return edge_index


def compute_edge_scores(embeddings, edge_index):
    """
    Compute similarity scores for graph edges.
    
    Used for graph reconstruction loss in the autoencoder.
    
    Args:
        embeddings: Node features (N, feature_dim)
        edge_index: Graph edges (2, num_edges)
        
    Returns:
        Edge similarity scores (num_edges,)
    
    Example:
        >>> embeddings = torch.randn(100, 128)
        >>> edge_index = build_knn_graph(embeddings, k=5)
        >>> scores = compute_edge_scores(embeddings, edge_index)
    """
    # Get source and target node embeddings
    src_emb = embeddings[edge_index[0]]
    tgt_emb = embeddings[edge_index[1]]
    
    # Compute cosine similarity
    scores = F.cosine_similarity(src_emb, tgt_emb, dim=1)
    
    return scores


def build_batch_knn_graph(embeddings, batch_indices, k=5):
    """
    Build k-NN graphs for batched data.
    
    Creates separate k-NN graphs for each sample in the batch,
    ensuring no edges cross batch boundaries.
    
    Args:
        embeddings: Embeddings (total_nodes, feature_dim)
        batch_indices: Batch assignment for each node (total_nodes,)
        k: Number of nearest neighbors
        
    Returns:
        edge_index: Combined edge index (2, total_edges)
    
    Example:
        >>> # Suppose we have 3 samples with 10 nodes each
        >>> embeddings = torch.randn(30, 128)
        >>> batch_indices = torch.tensor([0]*10 + [1]*10 + [2]*10)
        >>> edge_index = build_batch_knn_graph(embeddings, batch_indices, k=5)
    """
    edge_indices = []
    
    for batch_id in torch.unique(batch_indices):
        # Get nodes for this batch
        mask = batch_indices == batch_id
        batch_embeddings = embeddings[mask]
        
        # Build graph for this batch
        batch_edge_index = build_knn_graph(batch_embeddings, k=k)
        
        # Adjust indices to global indexing
        node_offset = torch.where(mask)[0][0]
        batch_edge_index = batch_edge_index + node_offset
        
        edge_indices.append(batch_edge_index)
    
    # Combine all edge indices
    edge_index = torch.cat(edge_indices, dim=1)
    
    return edge_index


def visualize_graph(embeddings, edge_index, labels=None, method='tsne'):
    """
    Visualize the k-NN graph structure.
    
    Args:
        embeddings: Node features (N, feature_dim)
        edge_index: Graph edges (2, num_edges)
        labels: Optional node labels for coloring
        method: Dimensionality reduction method ('tsne' or 'umap')
        
    Returns:
        matplotlib figure
    
    Example:
        >>> embeddings = torch.randn(100, 128)
        >>> edge_index = build_knn_graph(embeddings, k=5)
        >>> labels = torch.randint(0, 5, (100,))
        >>> fig = visualize_graph(embeddings, edge_index, labels)
    """
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    
    # Convert to numpy
    if isinstance(embeddings, torch.Tensor):
        embeddings_np = embeddings.detach().cpu().numpy()
    else:
        embeddings_np = embeddings
    
    # Dimensionality reduction
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
        coords = reducer.fit_transform(embeddings_np)
    elif method == 'umap':
        try:
            from umap import UMAP
            reducer = UMAP(n_components=2, random_state=42)
            coords = reducer.fit_transform(embeddings_np)
        except ImportError:
            print("UMAP not installed, falling back to t-SNE")
            reducer = TSNE(n_components=2, random_state=42)
            coords = reducer.fit_transform(embeddings_np)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot edges
    edge_index_np = edge_index.cpu().numpy()
    for i in range(edge_index_np.shape[1]):
        src, tgt = edge_index_np[:, i]
        ax.plot([coords[src, 0], coords[tgt, 0]], 
                [coords[src, 1], coords[tgt, 1]], 
                'gray', alpha=0.1, linewidth=0.5)
    
    # Plot nodes
    if labels is not None:
        if isinstance(labels, torch.Tensor):
            labels_np = labels.detach().cpu().numpy()
        else:
            labels_np = labels
        scatter = ax.scatter(coords[:, 0], coords[:, 1], 
                           c=labels_np, cmap='tab10', 
                           s=50, alpha=0.7, edgecolors='black', linewidths=0.5)
        plt.colorbar(scatter, ax=ax, label='Class')
    else:
        ax.scatter(coords[:, 0], coords[:, 1], 
                  s=50, alpha=0.7, edgecolors='black', linewidths=0.5)
    
    ax.set_title('k-NN Graph Structure', fontsize=16)
    ax.set_xlabel('Dimension 1', fontsize=12)
    ax.set_ylabel('Dimension 2', fontsize=12)
    plt.tight_layout()
    
    return fig
