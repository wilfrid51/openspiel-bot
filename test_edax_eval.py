"""
Test script for EdaxEvaluator
"""

from edax_eval import EdaxEvaluator, BLACK, WHITE, EMPTY

def print_board(board):
    """Print board in readable format."""
    print("  a b c d e f g h")
    for i, row in enumerate(board):
        print(f"{8-i} ", end="")
        for cell in row:
            if cell == EMPTY:
                print(". ", end="")
            elif cell == BLACK:
                print("X ", end="")
            else:
                print("O ", end="")
        print(f" {8-i}")
    print("  a b c d e f g h")

# Create evaluator
evaluator = EdaxEvaluator()

# Test with initial position
print("=== Initial Othello Position ===")
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
print_board(initial_board)

score_black = evaluator.evaluate(initial_board, BLACK)
score_white = evaluator.evaluate(initial_board, WHITE)
print(f"\nEvaluation (Black's perspective): {score_black:+d}")
print(f"Evaluation (White's perspective): {score_white:+d}")

# Test with a mid-game position
print("\n=== Mid-Game Position ===")
midgame_board = [
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 2, 0, 0, 0],
    [0, 0, 1, 1, 2, 0, 0, 0],
    [0, 0, 2, 2, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
]
print_board(midgame_board)

score_black = evaluator.evaluate(midgame_board, BLACK)
score_white = evaluator.evaluate(midgame_board, WHITE)
print(f"\nEvaluation (Black's perspective): {score_black:+d}")
print(f"Evaluation (White's perspective): {score_white:+d}")

# Show feature extraction
print("\n=== Feature Extraction Example ===")
features = evaluator.extract_features(initial_board, BLACK)
print(f"Total features extracted: {len(features)}")
print(f"Feature indices (first 20): {features[:20]}")

# Show raw accumulation
empty_count = sum(row.count(EMPTY) for row in initial_board)
ply = 60 - empty_count
raw_score = evaluator.eval_accumulate(features, ply, BLACK)
print(f"\nPly (game stage): {ply}")
print(f"Raw accumulated score: {raw_score}")
print(f"Normalized score: {score_black}")
