import numpy as np
import torch
import matplotlib.pyplot as plt
from metrics.context_fid import TS2Vec

def compute_memorization_metric(real_data, generated_data, device='cuda', plot_path=None):
    """
    Computes the cosine similarity between each generated sample and its nearest neighbor 
    in the training set (real_data) using TS2Vec embeddings.
    
    Args:
        real_data (np.ndarray): Training data (B, T, C)
        generated_data (np.ndarray): Generated data (B, T, C)
        device (str): 'cuda' or 'cpu'
        plot_path (str): Optional path to save the histogram plot
        
    Returns:
        dict: Statistics about the nearest neighbor similarities (mean, max, min, etc.)
    """
    print(f"Computing memorization metric...")
    print(f"Real data shape: {real_data.shape}")
    print(f"Generated data shape: {generated_data.shape}")

    # 1. Train/Fit TS2Vec encoder on real data to learn representations
    # We use the same settings as Context-FID
    input_dims = real_data.shape[-1]
    model = TS2Vec(
        input_dims=input_dims, 
        device=device, 
        batch_size=8, 
        lr=0.001, 
        output_dims=320,
        max_train_length=3000
    )
    
    print("Fitting TS2Vec encoder...")
    model.fit(real_data, verbose=False)
    
    # 2. Encode both datasets
    print("Encoding data...")
    real_emb = model.encode(real_data, encoding_window='full_series')
    gen_emb = model.encode(generated_data, encoding_window='full_series')
    
    # Convert to torch for fast matrix multiplication
    real_emb = torch.from_numpy(real_emb).to(device)
    gen_emb = torch.from_numpy(gen_emb).to(device)
    
    # Normalize embeddings to unit length for cosine similarity
    real_emb = torch.nn.functional.normalize(real_emb, p=2, dim=1)
    gen_emb = torch.nn.functional.normalize(gen_emb, p=2, dim=1)
    
    # 3. Compute Cosine Similarity Matrix
    # Shape: (Num_Gen, Num_Real)
    print("Computing similarity matrix...")
    similarity_matrix = torch.matmul(gen_emb, real_emb.T)
    
    # 4. Find nearest neighbor for each generated sample (max similarity)
    # Shape: (Num_Gen,)
    max_similarities, _ = torch.max(similarity_matrix, dim=1)
    max_similarities = max_similarities.cpu().numpy()
    
    # 5. Statistics
    stats = {
        'mean_similarity': np.mean(max_similarities),
        'median_similarity': np.median(max_similarities),
        'max_similarity': np.max(max_similarities), # Worst case memorization
        'min_similarity': np.min(max_similarities),
        'std_similarity': np.std(max_similarities),
        'percent_over_0.95': np.mean(max_similarities > 0.95) * 100, # Approx exact copies
        'percent_over_0.90': np.mean(max_similarities > 0.90) * 100
    }
    
    print(f"Memorization Stats: {stats}")
    
    # 6. Plotting
    if plot_path:
        plt.figure(figsize=(12, 7))
        plt.hist(max_similarities, bins=50, alpha=0.75, color='blue', edgecolor='black')
        plt.title('Distribution of Nearest Neighbor Similarities (Memorization Check)', fontsize=14, fontweight='bold')
        plt.xlabel('Cosine Similarity (TS2Vec Embeddings)', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.axvline(x=0.95, color='r', linestyle='--', label='Likely Copy (>0.95)', linewidth=2)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # Add statistics text box
        stats_text = (
            f"Mean: {stats['mean_similarity']:.4f}\n"
            f"Median: {stats['median_similarity']:.4f}\n"
            f"Max: {stats['max_similarity']:.4f}\n"
            f"Min: {stats['min_similarity']:.4f}\n"
            f"Std: {stats['std_similarity']:.4f}\n"
            f"% > 0.95: {stats['percent_over_0.95']:.2f}%\n"
            f"% > 0.90: {stats['percent_over_0.90']:.2f}%"
        )
        
        # Place text box in upper right corner
        plt.text(0.98, 0.98, stats_text, 
                transform=plt.gca().transAxes,
                fontsize=10,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                family='monospace')
        
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Histogram saved to {plot_path}")
        plt.close()
        
    return stats
