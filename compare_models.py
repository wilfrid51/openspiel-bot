#!/usr/bin/env python3
"""
Compare parameters of two models from Hugging Face repositories.
"""

import torch
from transformers import AutoModel, AutoConfig
import sys

def count_parameters(model):
    """Count total, trainable, and non-trainable parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': non_trainable_params
    }

def format_number(num):
    """Format large numbers with commas and units."""
    if num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.2f}B ({num:,})"
    elif num >= 1_000_000:
        return f"{num / 1_000_000:.2f}M ({num:,})"
    elif num >= 1_000:
        return f"{num / 1_000:.2f}K ({num:,})"
    else:
        return str(num)

def load_and_compare_models(repo1, repo2, device='cpu'):
    """
    Load two models from Hugging Face and compare their parameters.
    
    Args:
        repo1: First Hugging Face repository ID
        repo2: Second Hugging Face repository ID
        device: Device to load models on ('cpu' or 'cuda')
    """
    print(f"Loading models from Hugging Face...")
    print(f"Model 1: {repo1}")
    print(f"Model 2: {repo2}")
    print("=" * 80)
    
    try:
        # Load first model
        print(f"\n[1/2] Loading {repo1}...")
        config1 = AutoConfig.from_pretrained(repo1, trust_remote_code=True)
        model1 = AutoModel.from_pretrained(
            repo1, 
            trust_remote_code=True,
            torch_dtype=torch.float32,
            device_map=device if device == 'cpu' else 'auto'
        )
        params1 = count_parameters(model1)
        print(f"✓ Loaded {repo1}")
        
        # Load second model
        print(f"\n[2/2] Loading {repo2}...")
        config2 = AutoConfig.from_pretrained(repo2, trust_remote_code=True)
        model2 = AutoModel.from_pretrained(
            repo2,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            device_map=device if device == 'cpu' else 'auto'
        )
        params2 = count_parameters(model2)
        print(f"✓ Loaded {repo2}")
        
        # Print comparison
        print("\n" + "=" * 80)
        print("PARAMETER COMPARISON")
        print("=" * 80)
        
        print(f"\n{'Metric':<25} {'Model 1':<30} {'Model 2':<30} {'Difference':<20}")
        print("-" * 105)
        
        # Total parameters
        diff_total = params1['total'] - params2['total']
        print(f"{'Total Parameters':<25} {format_number(params1['total']):<30} {format_number(params2['total']):<30} {format_number(diff_total):<20}")
        
        # Trainable parameters
        diff_trainable = params1['trainable'] - params2['trainable']
        print(f"{'Trainable Parameters':<25} {format_number(params1['trainable']):<30} {format_number(params2['trainable']):<30} {format_number(diff_trainable):<20}")
        
        # Non-trainable parameters
        diff_non_trainable = params1['non_trainable'] - params2['non_trainable']
        print(f"{'Non-trainable Parameters':<25} {format_number(params1['non_trainable']):<30} {format_number(params2['non_trainable']):<30} {format_number(diff_non_trainable):<20}")
        
        # Ratio
        if params2['total'] > 0:
            ratio = params1['total'] / params2['total']
            print(f"\n{'Parameter Ratio (Model1/Model2)':<25} {ratio:.4f}x")
        
        # Model configurations
        print("\n" + "=" * 80)
        print("MODEL CONFIGURATIONS")
        print("=" * 80)
        
        print(f"\n{repo1}:")
        print(f"  Architecture: {config1.architectures[0] if hasattr(config1, 'architectures') and config1.architectures else 'N/A'}")
        print(f"  Hidden size: {getattr(config1, 'hidden_size', 'N/A')}")
        print(f"  Num layers: {getattr(config1, 'num_hidden_layers', getattr(config1, 'num_layers', 'N/A'))}")
        print(f"  Num attention heads: {getattr(config1, 'num_attention_heads', 'N/A')}")
        print(f"  Vocab size: {getattr(config1, 'vocab_size', 'N/A')}")
        
        print(f"\n{repo2}:")
        print(f"  Architecture: {config2.architectures[0] if hasattr(config2, 'architectures') and config2.architectures else 'N/A'}")
        print(f"  Hidden size: {getattr(config2, 'hidden_size', 'N/A')}")
        print(f"  Num layers: {getattr(config2, 'num_hidden_layers', getattr(config2, 'num_layers', 'N/A'))}")
        print(f"  Num attention heads: {getattr(config2, 'num_attention_heads', 'N/A')}")
        print(f"  Vocab size: {getattr(config2, 'vocab_size', 'N/A')}")
        
        # Clean up memory
        del model1, model2
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return params1, params2
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare parameters of two Hugging Face models')
    parser.add_argument('repo1', type=str, help='First Hugging Face repository ID')
    parser.add_argument('repo2', type=str, help='Second Hugging Face repository ID')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                        help='Device to load models on (default: cpu)')
    
    args = parser.parse_args()
    
    load_and_compare_models(args.repo1, args.repo2, device=args.device)
