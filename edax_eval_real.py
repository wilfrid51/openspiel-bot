"""
Edax evaluator using REAL Edax weights (simplified version).

NOTE: This uses Edax's real trained weights from eval.dat.
However, the full Edax evaluation includes complex symmetry unpacking.
This version uses a simplified approach that gives approximate results.

For exact Edax scores, you would need to:
1. Implement the full symmetry unpacking (complex)
2. Or call Edax directly via subprocess
"""

import numpy as np
from simple_edax_loader import load_edax_weights_simple
from typing import List

# Constants  
BLACK = 1
WHITE = 2
EMPTY = 0
SCORE_MIN = -64
SCORE_MAX = 64


class EdaxEvaluatorReal:
    """
    Edax evaluator using real weights (simplified).
    """
    
    def __init__(self, eval_file: str = "/root/workspace/openspiel-bot/edax-reversi/data/eval.dat"):
        """
        Initialize with real Edax weights.
        
        Args:
            eval_file: Path to Edax eval.dat file
        """
        print("Loading real Edax weights...")
        self.packed_weights, self.version_info = load_edax_weights_simple(eval_file)
        print(f"✓ Loaded Edax v{self.version_info['version']}.{self.version_info['release']}.{self.version_info['build']}")
        print("\nNOTE: This is a simplified evaluator.")
        print("Scores will be APPROXIMATE, not exact Edax scores.")
        print("For exact scores, use Edax hint() directly.\n")
    
    def evaluate_simple(self, board: List[List[int]], player: int) -> int:
        """
        Simple evaluation using a subset of Edax weights.
        
        This is a VERY simplified version that uses some of the packed weights
        directly without full feature extraction. It's meant as a demonstration.
        
        For real use, you should either:
        1. Implement full Edax feature extraction + unpacking
        2. Call Edax directly via subprocess
        
        Args:
            board: 8x8 board
            player: Current player (BLACK=1 or WHITE=2)
            
        Returns:
            Approximate evaluation score
        """
        # Count pieces and empty squares
        empty_count = sum(row.count(EMPTY) for row in board)
        ply = min(60 - empty_count, 60)  # Clamp to valid ply range
        
        player_idx = 0 if player == BLACK else 1
        
        # Simple heuristic using piece counts and positions
        # This is NOT how Edax actually evaluates, but demonstrates using the weights
        score = 0
        
        # Weight by position (using first few packed weights as positional bonuses)
        for y in range(8):
            for x in range(8):
                cell = board[y][x]
                if cell != EMPTY:
                    # Use position as a simple index into weights
                    idx = min(y * 8 + x, len(self.packed_weights[ply]) - 1)
                    weight = int(self.packed_weights[ply, idx])
                    
                    if cell == player:
                        score += weight // 10  # Scale down
                    else:
                        score -= weight // 10
        
        # Normalize to Edax-like range
        score = score // 128
        score = max(SCORE_MIN + 1, min(SCORE_MAX - 1, score))
        
        return score


def main():
    """Demo of real Edax weights."""
    eval = EdaxEvaluatorReal()
    
    # Test with initial position
    initial_board = [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 2, 0, 0, 0],
        [0, 0, 0, 2, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ]
    
    score_black = eval.evaluate_simple(initial_board, BLACK)
    score_white = eval.evaluate_simple(initial_board, WHITE)
    
    print(f"Initial position evaluation:")
    print(f"  Black: {score_black:+d}")
    print(f"  White: {score_white:+d}")
    
    print(f"\nWeight statistics (ply 0):")
    print(f"  Min: {eval.packed_weights[0].min()}")
    print(f"  Max: {eval.packed_weights[0].max()}")
    print(f"  Mean: {eval.packed_weights[0].mean():.1f}")
    print(f"  Std: {eval.packed_weights[0].std():.1f}")
    
    print(f"\nWeight statistics (ply 30):")
    print(f"  Min: {eval.packed_weights[30].min()}")
    print(f"  Max: {eval.packed_weights[30].max()}")
    print(f"  Mean: {eval.packed_weights[30].mean():.1f}")
    print(f"  Std: {eval.packed_weights[30].std():.1f}")


if __name__ == "__main__":
    main()
