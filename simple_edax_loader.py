"""
Simplified Edax weight loader - loads weights directly without complex unpacking.
This version reads the packed weights and expands them using a simpler method.
"""

import struct
import numpy as np
from pathlib import Path

# Constants
EVAL_N_PLY = 61
N_PACKED_WEIGHTS = 114364

def load_edax_weights_simple(eval_file: str) -> tuple:
    """
    Load Edax packed weights.
    Returns (packed_weights, version_info) where:
    - packed_weights: shape (61, 114364) - the raw packed weights per ply
    - version_info: dict with version details
    """
    eval_path = Path(eval_file)
    if not eval_path.exists():
        raise FileNotFoundError(f"Evaluation file not found: {eval_file}")
    
    print(f"Loading Edax packed weights from {eval_file}...")
    
    packed_weights = np.zeros((EVAL_N_PLY, N_PACKED_WEIGHTS), dtype=np.int16)
    
    with open(eval_file, 'rb') as f:
        # Read header
        edax_header = struct.unpack('<I', f.read(4))[0]
        eval_header = struct.unpack('<I', f.read(4))[0]
        
        # Check if headers are big-endian (XADE/LAVE)
        is_header_swapped = (edax_header == 0x45444158 and eval_header == 0x4556414C)
        
        # Note: Even if headers are big-endian, version and weights are ALWAYS little-endian
        # (This is how Edax stores the file)
        version = struct.unpack('<I', f.read(4))[0]
        release = struct.unpack('<I', f.read(4))[0]
        build = struct.unpack('<I', f.read(4))[0]
        date = struct.unpack('<d', f.read(8))[0]
        
        version_info = {
            'version': version,
            'release': release,
            'build': build,
            'date': date,
            'header_swapped': is_header_swapped
        }
        
        print(f"Version: {version}.{release}.{build}")
        print(f"Header format: {'big-endian (XADE/LAVE)' if is_header_swapped else 'little-endian (EDAX/EVAL)'}")
        
        # Read packed weights for each ply
        for ply in range(EVAL_N_PLY):
            if ply % 10 == 0:
                print(f"Loading ply {ply}/{EVAL_N_PLY}...")
            
            # Read packed weights as little-endian int16
            weights_bytes = f.read(N_PACKED_WEIGHTS * 2)
            packed_weights[ply] = np.frombuffer(weights_bytes, dtype='<i2')
    
    print(f"✓ Successfully loaded {EVAL_N_PLY} plies of packed weights")
    print(f"  Shape: {packed_weights.shape}")
    print(f"  Weight range: [{packed_weights.min()}, {packed_weights.max()}]")
    
    return packed_weights, version_info


if __name__ == "__main__":
    eval_file = "/root/workspace/openspiel-bot/edax-reversi/data/eval.dat"
    weights, info = load_edax_weights_simple(eval_file)
    
    print(f"\nSample weights (ply 0):")
    print(f"  First 20: {weights[0, :20]}")
    print(f"  Last 20: {weights[0, -20:]}")
    print(f"\nSample weights (ply 30):")
    print(f"  First 20: {weights[30, :20]}")
