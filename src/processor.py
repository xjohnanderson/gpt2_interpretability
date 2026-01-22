import torch
import numpy as np
from collections import defaultdict

class MagnitudeProcessor:
    def __init__(self):
        """
        Initializes the processor with nested dictionaries to store sums and counts
        for Attention and MLP sub-layers.
        """
        # Structure: [layer_idx][pos_tag] -> {'sum': float, 'count': int}
        self.attn_stats = defaultdict(lambda: defaultdict(lambda: {'sum': 0.0, 'count': 0}))
        self.mlp_stats = defaultdict(lambda: defaultdict(lambda: {'sum': 0.0, 'count': 0}))

    def process_sequence(self, layer_idx, pos_tags, attn_res, mlp_res):
        """
        Calculates L2 norms for a single layer's residuals and aggregates by POS tag.
        
        Args:
            layer_idx (int): The current Transformer layer index.
            pos_tags (list): Aligned POS tags for the sequence.
            attn_res (torch.Tensor): Attention residual tensor (seq_len, hidden_dim).
            mlp_res (torch.Tensor): MLP residual tensor (seq_len, hidden_dim).
        """
        # Calculate L2 Norms: sqrt(sum(x^2))
        # Resulting shape: (seq_len,)
        attn_norms = torch.linalg.norm(attn_res, dim=-1).cpu().numpy()
        mlp_norms = torch.linalg.norm(mlp_res, dim=-1).cpu().numpy()

        for i, tag in enumerate(pos_tags):
            if tag == 'UNKNOWN':
                continue
            
            # Update Attention Statistics
            self.attn_stats[layer_idx][tag]['sum'] += attn_norms[i]
            self.attn_stats[layer_idx][tag]['count'] += 1
            
            # Update MLP Statistics
            self.mlp_stats[layer_idx][tag]['sum'] += mlp_norms[i]
            self.mlp_stats[layer_idx][tag]['count'] += 1

    def get_means(self, mode='attn'):
        """
        Computes the final average magnitudes for plotting.
        
        Args:
            mode (str): 'attn' for Attention residuals, 'mlp' for MLP residuals.
        Returns:
            dict: Layer -> POS -> Mean Magnitude
        """
        target_dict = self.attn_stats if mode == 'attn' else self.mlp_stats
        means_dict = defaultdict(dict)
        
        for layer, tags in target_dict.items():
            for tag, data in tags.items():
                if data['count'] > 0:
                    means_dict[layer][tag] = data['sum'] / data['count']
        
        return means_dict