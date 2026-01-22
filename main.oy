import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from collections import defaultdict
import matplotlib.pyplot as plt
import os

# Import our modular components
from src.model_utils import load_gpt2_and_tokenizer
from src.hooks import ResidualHookManager
from src.alignment import align_pos_to_tokens
import spacy

def main():
    # 1. Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model, tokenizer = load_gpt2_and_tokenizer()
    model.to(device).eval()
    
    nlp = spacy.load('en_core_web_sm')
    hook_manager = ResidualHookManager(model)
    hook_manager.register_hooks()

    # 2. Data Preparation
    print("Loading and preprocessing dataset...")
    raw_datasets = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train[:1%]') # Sampled for speed
    
    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, max_length=512)

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    data_loader = DataLoader(tokenized_datasets, batch_size=8)

    # 3. Aggregation Dicts
    # Structure: [layer_idx][pos_tag] -> {'sum': val, 'count': val}
    attn_stats = defaultdict(lambda: defaultdict(lambda: {'sum': 0.0, 'count': 0}))
    mlp_stats = defaultdict(lambda: defaultdict(lambda: {'sum': 0.0, 'count': 0}))

    # 4. Processing Loop
    print("Starting extraction...")
    for batch_idx, batch in enumerate(data_loader):
        hook_manager.clear()
        
        input_ids = torch.stack(batch['input_ids']).to(device)
        attn_mask = torch.stack(batch['attention_mask']).to(device)

        with torch.no_grad():
            model(input_ids, attention_mask=attn_mask)

        # Iterate through sequences in batch
        for seq_idx in range(input_ids.shape[0]):
            # Get valid tokens (masking padding)
            valid_len = attn_mask[seq_idx].sum().item()
            token_ids = input_ids[seq_idx, :valid_len].tolist()
            hf_tokens = [tokenizer.decode([t]) for t in token_ids]
            
            # Align POS tags
            decoded_text = tokenizer.decode(token_ids, skip_special_tokens=True)
            doc = nlp(decoded_text)
            aligned_tags = align_pos_to_tokens(doc, hf_tokens, tokenizer)

            # Process Layers
            for layer_idx in range(len(model.h)):
                # Extract and calculate Norms
                attn_res = hook_manager.attn_outputs[layer_idx][seq_idx, :valid_len, :]
                mlp_res = hook_manager.mlp_outputs[layer_idx][seq_idx, :valid_len, :]
                
                attn_norms = torch.linalg.norm(attn_res, dim=-1).cpu().numpy()
                mlp_norms = torch.linalg.norm(mlp_res, dim=-1).cpu().numpy()

                for i, tag in enumerate(aligned_tags):
                    if tag == 'UNKNOWN': continue
                    
                    attn_stats[layer_idx][tag]['sum'] += attn_norms[i]
                    attn_stats[layer_idx][tag]['count'] += 1
                    
                    mlp_stats[layer_idx][tag]['sum'] += mlp_norms[i]
                    mlp_stats[layer_idx][tag]['count'] += 1

        if (batch_idx + 1) % 5 == 0:
            print(f"Processed batch {batch_idx + 1}")

    hook_manager.remove_hooks()

    # 5. Visualization
    plot_results(attn_stats, "Attention Residual Magnitudes", "attn_results.png")
    plot_results(mlp_stats, "MLP Residual Magnitudes", "mlp_results.png")

def plot_results(stats, title, filename):
    plt.figure(figsize=(12, 6))
    layers = sorted(stats.keys())
    all_tags = sorted(list(set(tag for l in stats.values() for tag in l.keys())))

    for tag in all_tags:
        means = []
        for l in layers:
            data = stats[l].get(tag, {'sum': 0, 'count': 1})
            means.append(data['sum'] / max(data['count'], 1))
        
        if sum(means) > 0:
            plt.plot(layers, means, label=tag, marker='o', markersize=4, alpha=0.7)

    plt.title(title)
    plt.xlabel("Layer Index")
    plt.ylabel("Mean L2 Norm")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    os.makedirs('results', exist_ok=True)
    plt.savefig(f"results/{filename}")
    print(f"Saved plot to results/{filename}")
    plt.show()

if __name__ == "__main__":
    main()