"""
Load Edax evaluation weights from binary file.
"""

import struct
import numpy as np
from pathlib import Path

# Constants from Edax eval.c
EVAL_N_PLY = 61
EVAL_N_WEIGHT = 226315
N_PACKED_WEIGHTS = 114364

# Magic numbers for file header
EDAX_MAGIC = 0x58414445  # "EDAX" in little-endian
EVAL_MAGIC = 0x4C415645  # "EVAL" in little-endian
XADE_MAGIC = 0x45444158  # "XADE" in big-endian (swapped)
LAVE_MAGIC = 0x4556414C  # "LAVE" in big-endian (swapped)

# Feature sizes (from eval.c)
EVAL_SIZE = [19683, 59049, 59049, 59049, 6561, 6561, 6561, 6561, 2187, 729, 243, 81, 1]
EVAL_PACKED_SIZE = [10206, 29889, 29646, 29646, 3321, 3321, 3321, 3321, 1134, 378, 135, 45, 1]

# Symmetry patterns
SYM_S10 = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
SYM_C10 = [9, 8, 7, 6, 4, 5, 3, 2, 1, 0]
SYM_C9 = [0, 2, 1, 4, 3, 5, 7, 6, 8]


def swap_bytes_16(value):
    """Swap bytes for 16-bit value (big-endian to little-endian)."""
    # Handle numpy int16
    if isinstance(value, np.int16):
        # Convert to unsigned for bit operations
        unsigned = np.uint16(value)
        swapped = ((unsigned & 0xFF) << 8) | ((unsigned & 0xFF00) >> 8)
        return np.int16(swapped)
    # Handle regular Python int
    unsigned = value & 0xFFFF
    swapped = ((unsigned & 0xFF) << 8) | ((unsigned & 0xFF00) >> 8)
    # Convert back to signed
    if swapped >= 0x8000:
        return swapped - 0x10000
    return swapped


def swap_bytes_32(value):
    """Swap bytes for 32-bit value (big-endian to little-endian)."""
    return ((value & 0xFF) << 24) | ((value & 0xFF00) << 8) | \
           ((value & 0xFF0000) >> 8) | ((value & 0xFF000000) >> 24)


def player_feature(sym, length, index):
    """Get player feature index from symmetry."""
    p = 0
    for i in range(length):
        p = p * 3 + ((index // (3 ** sym[i])) % 3)
    return p


def opponent_feature(index, length):
    """Get opponent feature index."""
    p = 0
    for i in range(length):
        c = index % 3
        if c == 1:
            c = 2
        elif c == 2:
            c = 1
        p = p * 3 + c
        index //= 3
    return p


def create_unpack_table(length, size, sym):
    """
    Create unpacking table for symmetry.
    Returns: (pack_player, pack_opponent) arrays
    """
    pack_player = np.zeros(size, dtype=np.int32)
    pack_opponent = np.zeros(size, dtype=np.int32)
    
    n = 0
    for i in range(size):
        j = player_feature(sym, length, i)
        if j < i:
            pack_player[i] = pack_player[j]
        else:
            pack_player[i] = n
            n += 1
        pack_opponent[opponent_feature(i, length)] = pack_player[i]
    
    return pack_player, pack_opponent


def load_edax_weights(eval_file: str) -> np.ndarray:
    """
    Load Edax evaluation weights from binary file.
    
    Args:
        eval_file: Path to Edax eval.dat file
        
    Returns:
        numpy array of shape (65, 2, EVAL_N_WEIGHT) containing weights
        [ply, player, feature_index]
    """
    eval_path = Path(eval_file)
    if not eval_path.exists():
        raise FileNotFoundError(f"Evaluation file not found: {eval_file}")
    
    print(f"Loading Edax weights from {eval_file}...")
    
    # Create unpacking tables
    print("Creating unpacking tables...")
    EVAL_S10_p, EVAL_S10_o = create_unpack_table(10, 59049, SYM_S10)
    EVAL_S8_p, EVAL_S8_o = create_unpack_table(8, 6561, SYM_S10[2:])
    EVAL_S7_p, EVAL_S7_o = create_unpack_table(7, 2187, SYM_S10[3:])
    EVAL_S6_p, EVAL_S6_o = create_unpack_table(6, 729, SYM_S10[4:])
    EVAL_S5_p, EVAL_S5_o = create_unpack_table(5, 243, SYM_S10[5:])
    EVAL_S4_p, EVAL_S4_o = create_unpack_table(4, 81, SYM_S10[6:])
    EVAL_C9_p, EVAL_C9_o = create_unpack_table(9, 19683, SYM_C9)
    EVAL_C10_p, EVAL_C10_o = create_unpack_table(10, 59049, SYM_C10)
    
    # Allocate weight array (65 plies to match original, even though only 61 are used)
    weights = np.zeros((65, 2, EVAL_N_WEIGHT + 1), dtype=np.int16)
    
    with open(eval_file, 'rb') as f:
        # Read header
        edax_header = struct.unpack('<I', f.read(4))[0]
        eval_header = struct.unpack('<I', f.read(4))[0]
        
        # Check magic numbers
        is_swapped = False
        if edax_header == XADE_MAGIC and eval_header == LAVE_MAGIC:
            is_swapped = True
        elif not (edax_header == EDAX_MAGIC and eval_header == EVAL_MAGIC):
            raise ValueError(f"Invalid Edax eval file format (headers: {edax_header:08X} {eval_header:08X})")
        
        # Read version info
        version = struct.unpack('<I', f.read(4))[0]
        release = struct.unpack('<I', f.read(4))[0]
        build = struct.unpack('<I', f.read(4))[0]
        date = struct.unpack('<d', f.read(8))[0]
        
        if is_swapped:
            version = swap_bytes_32(version)
            release = swap_bytes_32(release)
            build = swap_bytes_32(build)
        
        print(f"Edax eval file version {version}.{release}.{build} (date: {date})")
        
        # Read weights for each ply
        for ply in range(EVAL_N_PLY):
            if ply % 10 == 0:
                print(f"Loading ply {ply}/{EVAL_N_PLY}...")
            
            # Read packed weights
            packed_weights = np.frombuffer(f.read(N_PACKED_WEIGHTS * 2), dtype=np.int16)
            
            if is_swapped:
                # Swap bytes if needed
                packed_weights = np.array([swap_bytes_16(w) for w in packed_weights], dtype=np.int16)
            
            # Unpack weights for player 0 and player 1 (matching Edax eval.c lines 642-719)
            j = 0
            offset = 0
            
            # Feature 0: Corner 9 (EVAL_C9)
            for k in range(EVAL_SIZE[0]):
                weights[ply, 0, j] = packed_weights[EVAL_C9_p[k] + offset]
                weights[ply, 1, j] = packed_weights[EVAL_C9_o[k] + offset]
                j += 1
            offset += EVAL_PACKED_SIZE[0]
            
            # Feature 1: Corner 10 (EVAL_C10)
            for k in range(EVAL_SIZE[1]):
                weights[ply, 0, j] = packed_weights[EVAL_C10_p[k] + offset]
                weights[ply, 1, j] = packed_weights[EVAL_C10_o[k] + offset]
                j += 1
            offset += EVAL_PACKED_SIZE[1]
            
            # Feature 2: S10
            for k in range(EVAL_SIZE[2]):
                idx_p = EVAL_S10_p[k] + offset
                idx_o = EVAL_S10_o[k] + offset
                if ply == 0 and k < 5:
                    print(f"  Feature 2, k={k}: offset={offset}, EVAL_S10_p[k]={EVAL_S10_p[k]}, idx_p={idx_p}, packed_size={len(packed_weights)}")
                weights[ply, 0, j] = packed_weights[idx_p]
                weights[ply, 1, j] = packed_weights[idx_o]
                j += 1
            offset += EVAL_PACKED_SIZE[2]
            
            # Feature 3: S10
            for k in range(EVAL_SIZE[3]):
                weights[ply, 0, j] = packed_weights[EVAL_S10_p[k] + offset]
                weights[ply, 1, j] = packed_weights[EVAL_S10_o[k] + offset]
                j += 1
            offset += EVAL_PACKED_SIZE[3]
            
            # Features 4-7: S8
            for feat in range(4, 8):
                for k in range(EVAL_SIZE[feat]):
                    weights[ply, 0, j] = packed_weights[EVAL_S8_p[k] + offset]
                    weights[ply, 1, j] = packed_weights[EVAL_S8_o[k] + offset]
                    j += 1
                offset += EVAL_PACKED_SIZE[feat]
            
            # Feature 8: S7
            for k in range(EVAL_SIZE[8]):
                weights[ply, 0, j] = packed_weights[EVAL_S7_p[k] + offset]
                weights[ply, 1, j] = packed_weights[EVAL_S7_o[k] + offset]
                j += 1
            offset += EVAL_PACKED_SIZE[8]
            
            # Feature 9: S6
            for k in range(EVAL_SIZE[9]):
                weights[ply, 0, j] = packed_weights[EVAL_S6_p[k] + offset]
                weights[ply, 1, j] = packed_weights[EVAL_S6_o[k] + offset]
                j += 1
            offset += EVAL_PACKED_SIZE[9]
            
            # Feature 10: S5
            for k in range(EVAL_SIZE[10]):
                weights[ply, 0, j] = packed_weights[EVAL_S5_p[k] + offset]
                weights[ply, 1, j] = packed_weights[EVAL_S5_o[k] + offset]
                j += 1
            offset += EVAL_PACKED_SIZE[10]
            
            # Feature 11: S4
            for k in range(EVAL_SIZE[11]):
                weights[ply, 0, j] = packed_weights[EVAL_S4_p[k] + offset]
                weights[ply, 1, j] = packed_weights[EVAL_S4_o[k] + offset]
                j += 1
            offset += EVAL_PACKED_SIZE[11]
            
            # Feature 12: Final weight
            weights[ply, 0, j] = packed_weights[offset]
            weights[ply, 1, j] = packed_weights[offset]
    
    print(f"✓ Successfully loaded {EVAL_N_PLY} plies of weights")
    print(f"  Weight array shape: {weights.shape}")
    print(f"  Weight range: [{weights.min()}, {weights.max()}]")
    
    return weights


if __name__ == "__main__":
    # Test loading
    eval_file = "/root/workspace/openspiel-bot/edax-reversi/data/eval.dat"
    weights = load_edax_weights(eval_file)
    
    print(f"\nTest: Sample weights for ply 0, player 0:")
    print(f"  First 20 weights: {weights[0, 0, :20]}")
    print(f"  Last 20 weights: {weights[0, 0, -20:]}")
