"""
Python implementation of Edax's eval_accumulate() function.
Evaluates Othello positions using feature-based pattern matching.
"""

import numpy as np
from typing import List, Tuple, Optional

# Constants
BLACK = 1
WHITE = 2
EMPTY = 0

# Score bounds
SCORE_MIN = -64
SCORE_MAX = 64

# Weight offsets (from Edax eval.c)
WEIGHT_OFFSET = [0, 19683, 78732, 137781, 196830]

# Feature patterns: (n_squares, [square_coordinates])
# Coordinates: a1=0, b1=1, ..., h8=63
# Using standard Othello notation: a1-h8
EVAL_F2X = [
    # Corner patterns (9 squares)
    (9, [0, 1, 8, 9, 2, 16, 10, 17, 18]),      # A1 corner
    (9, [7, 6, 15, 14, 5, 23, 13, 22, 21]),    # H1 corner
    (9, [56, 57, 48, 49, 58, 40, 50, 41, 42]),  # A8 corner
    (9, [63, 62, 55, 54, 61, 47, 53, 46, 45]), # H8 corner
    
    # Edge patterns (10 squares)
    (10, [32, 33, 34, 35, 36, 37, 38, 39, 40, 41]),  # A column
    (10, [31, 30, 29, 28, 27, 26, 25, 24, 23, 22]),  # H column
    (10, [24, 25, 26, 27, 28, 29, 30, 31, 32, 33]),  # A row (bottom)
    (10, [39, 38, 37, 36, 35, 34, 33, 32, 31, 30]),  # H row (top)
    
    (10, [9, 0, 1, 2, 3, 4, 5, 6, 7, 14]),     # B2-H1 edge
    (10, [54, 56, 57, 58, 59, 60, 61, 62, 63, 55]),  # B7-A8 edge
    (10, [9, 8, 16, 24, 32, 40, 48, 56, 57, 54]),    # B2-A8 edge
    (10, [14, 15, 23, 31, 39, 47, 55, 63, 62, 55]),  # G2-H8 edge
    
    (10, [0, 2, 3, 10, 11, 18, 19, 4, 5, 7]),  # A1-H1 pattern
    (10, [56, 58, 59, 50, 51, 42, 43, 60, 61, 63]),  # A8-H8 pattern
    (10, [0, 16, 17, 25, 26, 34, 35, 24, 32, 56]),   # A1-A8 pattern
    (10, [7, 15, 14, 22, 21, 29, 28, 23, 31, 63]),   # H1-H8 pattern
    
    # Row patterns (8 squares)
    (8, [8, 9, 10, 11, 12, 13, 14, 15]),        # Row 2
    (8, [48, 49, 50, 51, 52, 53, 54, 55]),      # Row 7
    (8, [1, 9, 17, 25, 33, 41, 49, 57]),        # Column B
    (8, [6, 14, 22, 30, 38, 46, 54, 62]),       # Column G
    
    (8, [16, 17, 18, 19, 20, 21, 22, 23]),      # Row 3
    (8, [40, 41, 42, 43, 44, 45, 46, 47]),      # Row 6
    (8, [2, 10, 18, 26, 34, 42, 50, 58]),       # Column C
    (8, [5, 13, 21, 29, 37, 45, 53, 61]),       # Column F
    
    (8, [24, 25, 26, 27, 28, 29, 30, 31]),      # Row 4
    (8, [32, 33, 34, 35, 36, 37, 38, 39]),      # Row 5
    (8, [3, 11, 19, 27, 35, 43, 51, 59]),       # Column D
    (8, [4, 12, 20, 28, 36, 44, 52, 60]),       # Column E
    
    # Diagonal patterns (8 squares)
    (8, [0, 9, 18, 27, 36, 45, 54, 63]),        # Main diagonal
    (8, [7, 14, 21, 28, 35, 42, 49, 56]),       # Anti-diagonal
    
    # Diagonal patterns (7 squares)
    (7, [1, 10, 19, 28, 37, 46, 55]),           # Diagonal near main
    (7, [15, 22, 29, 36, 43, 50, 57]),          # Diagonal near anti
    (7, [8, 17, 26, 35, 44, 53, 62]),           # Diagonal near main
    (7, [6, 13, 20, 27, 34, 41, 48]),           # Diagonal near anti
    
    # Diagonal patterns (6 squares)
    (6, [2, 11, 20, 29, 38, 47]),               # Diagonal
    (6, [16, 25, 34, 43, 52, 61]),              # Diagonal
    (6, [5, 12, 19, 26, 33, 40]),               # Diagonal
    (6, [23, 30, 37, 44, 51, 58]),              # Diagonal
    
    # Diagonal patterns (5 squares)
    (5, [3, 12, 21, 30, 39]),                   # Diagonal
    (5, [24, 33, 42, 51, 60]),                  # Diagonal
    (5, [4, 11, 18, 25, 32]),                   # Diagonal
    (5, [31, 38, 45, 52, 59]),                  # Diagonal
    
    # Diagonal patterns (4 squares)
    (4, [3, 10, 17, 24]),                       # Diagonal
    (4, [32, 41, 50, 59]),                      # Diagonal
    (4, [4, 13, 22, 31]),                      # Diagonal
    (4, [39, 46, 53, 60]),                     # Diagonal
    
    # Empty patterns (for padding)
    (0, []),
    (0, []),
]

# Feature offsets (from Edax eval.c)
FEATURE_OFFSET = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    6561, 6561, 6561, 6561,
    13122, 13122, 13122, 13122,
    19683, 19683,
    26244, 26244, 26244, 26244,
    28431, 28431, 28431, 28431,
    29160, 29160, 29160, 29160,
    29403, 29403, 29403, 29403,
    29484, 29485
]


class EdaxEvaluator:
    """
    Python implementation of Edax's evaluation system.
    """
    
    def __init__(self, weights: Optional[np.ndarray] = None):
        """
        Initialize evaluator.
        
        Args:
            weights: Optional pre-loaded weights array.
                     Shape: (65, 2, N) where N is total weight count.
                     If None, creates dummy weights for demonstration.
        """
        if weights is None:
            # Create dummy weights for demonstration
            # In real Edax, these are loaded from a file
            self.weights = self._create_dummy_weights()
        else:
            self.weights = weights
    
    def _create_dummy_weights(self) -> np.ndarray:
        """Create dummy weights for demonstration purposes."""
        # Total weight count from Edax: EVAL_N_WEIGHT = 226315
        # For demo, we'll use smaller arrays
        # Real weights would be loaded from Edax's .eval file
        total_weights = 300000  # Approximate size
        weights = np.zeros((65, 2, total_weights), dtype=np.int16)
        
        # Initialize with small random values for demonstration
        # In real Edax, these are learned/tuned weights
        np.random.seed(42)
        weights = np.random.randint(-100, 100, size=(65, 2, total_weights), dtype=np.int16)
        
        return weights
    
    def board_to_array(self, board: List[List[int]]) -> np.ndarray:
        """
        Convert 2D board to 1D array.
        
        Args:
            board: 8x8 board, 0=empty, 1=black, 2=white
            
        Returns:
            1D array of 64 squares
        """
        arr = np.zeros(64, dtype=np.int8)
        for y in range(8):
            for x in range(8):
                idx = y * 8 + x
                arr[idx] = board[y][x]
        return arr
    
    def extract_features(self, board: List[List[int]], player: int) -> List[int]:
        """
        Extract features from board position.
        
        Args:
            board: 8x8 board representation
            player: Current player (BLACK=1 or WHITE=2)
            
        Returns:
            List of 47 feature indices
        """
        board_arr = self.board_to_array(board)
        features = []
        
        for i, (n_squares, squares) in enumerate(EVAL_F2X):
            if n_squares == 0:
                features.append(0)
                continue
            
            # Encode pattern as base-3 number
            feature_value = 0
            for square in squares:
                if square >= len(board_arr):
                    # Handle edge cases
                    feature_value = feature_value * 3 + EMPTY
                else:
                    color = board_arr[square]
                    # Normalize: 0=empty, 1=current_player, 2=opponent
                    if color == EMPTY:
                        normalized = 0
                    elif color == player:
                        normalized = 1
                    else:
                        normalized = 2
                    feature_value = feature_value * 3 + normalized
            
            # Add feature offset
            feature_value += FEATURE_OFFSET[i] if i < len(FEATURE_OFFSET) else 0
            features.append(feature_value)
        
        return features
    
    def eval_accumulate(self, features: List[int], ply: int, player: int) -> int:
        """
        Accumulate evaluation score from features.
        
        This is the Python equivalent of Edax's eval_accumulate() function.
        
        Args:
            features: List of 47 feature indices
            ply: Game stage (0-60 empty squares, or 60 - empty_count)
            player: Current player (0=black, 1=white)
            
        Returns:
            Raw evaluation score (before normalization)
        """
        # Clamp ply to valid range
        ply = max(0, min(64, ply))
        player_idx = 0 if player == BLACK else 1
        
        # Get weight arrays
        w0 = self.weights[ply, player_idx]
        w1 = w0[WEIGHT_OFFSET[1]:]
        w2 = w0[WEIGHT_OFFSET[2]:]
        w3 = w0[WEIGHT_OFFSET[3]:]
        w4 = w0[WEIGHT_OFFSET[4]:]
        
        # Accumulate weights (matching Edax eval_accumulate structure)
        sum_score = (
            w0[features[0]] + w0[features[1]] + w0[features[2]] + w0[features[3]] +
            w1[features[4]] + w1[features[5]] + w1[features[6]] + w1[features[7]] +
            w2[features[8]] + w2[features[9]] + w2[features[10]] + w2[features[11]] +
            w3[features[12]] + w3[features[13]] + w3[features[14]] + w3[features[15]] +
            w4[features[16]] + w4[features[17]] + w4[features[18]] + w4[features[19]] +
            w4[features[20]] + w4[features[21]] + w4[features[22]] + w4[features[23]] +
            w4[features[24]] + w4[features[25]] + w4[features[26]] + w4[features[27]] +
            w4[features[28]] + w4[features[29]] +
            w4[features[30]] + w4[features[31]] + w4[features[32]] + w4[features[33]] +
            w4[features[34]] + w4[features[35]] + w4[features[36]] + w4[features[37]] +
            w4[features[38]] + w4[features[39]] + w4[features[40]] + w4[features[41]] +
            w4[features[42]] + w4[features[43]] + w4[features[44]] + w4[features[45]] +
            w4[features[46]]
        )
        
        return int(sum_score)
    
    def search_eval_0(self, board: List[List[int]], player: int) -> int:
        """
        Evaluate position at depth 0 (static evaluation).
        
        This is the Python equivalent of Edax's search_eval_0() function.
        
        Args:
            board: 8x8 board representation
            player: Current player (BLACK=1 or WHITE=2)
            
        Returns:
            Normalized score in range [SCORE_MIN+1, SCORE_MAX-1]
        """
        # Count empty squares to determine ply
        empty_count = sum(row.count(EMPTY) for row in board)
        ply = 60 - empty_count
        
        # Extract features
        features = self.extract_features(board, player)
        
        # Accumulate raw score
        score = self.eval_accumulate(features, ply, player)
        
        # Normalize (matching Edax's normalization)
        if score > 0:
            score += 64
        else:
            score -= 64
        score //= 128
        
        # Clamp to valid range
        if score <= SCORE_MIN:
            score = SCORE_MIN + 1
        elif score >= SCORE_MAX:
            score = SCORE_MAX - 1
        
        return score
    
    def evaluate(self, board: List[List[int]], player: int) -> int:
        """
        Evaluate a position from the current player's perspective.
        
        Args:
            board: 8x8 board, 0=empty, 1=black, 2=white
            player: Current player (BLACK=1 or WHITE=2)
            
        Returns:
            Evaluation score: positive = good for player, negative = bad
        """
        return self.search_eval_0(board, player)


# Example usage
if __name__ == "__main__":
    # Create evaluator
    evaluator = EdaxEvaluator()
    
    # Example: Initial Othello position
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
    
    # Evaluate from black's perspective
    score_black = evaluator.evaluate(initial_board, BLACK)
    print(f"Evaluation (Black's perspective): {score_black}")
    
    # Evaluate from white's perspective
    score_white = evaluator.evaluate(initial_board, WHITE)
    print(f"Evaluation (White's perspective): {score_white}")
    
    # Extract features
    features = evaluator.extract_features(initial_board, BLACK)
    print(f"\nExtracted {len(features)} features")
    print(f"First 10 features: {features[:10]}")
    
    # Raw accumulation
    empty_count = sum(row.count(EMPTY) for row in initial_board)
    ply = 60 - empty_count
    raw_score = evaluator.eval_accumulate(features, ply, BLACK)
    print(f"\nRaw accumulated score: {raw_score}")
    print(f"Normalized score: {score_black}")
