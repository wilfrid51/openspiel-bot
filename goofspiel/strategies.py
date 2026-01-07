from .base import GSStrategy
import random


class RandomStrategy(GSStrategy):
    """Chooses a random valid bid each turn."""
    name = "random"
    def __init__(self, game_params):
        self.n = game_params['length']

    def start_game(self):
        self.valid_moves = list(range(self.n))
        random.shuffle(self.valid_moves)

    def get_bid(self, turn_value):
        return self.valid_moves.pop()

    def update_history(self, hist):
        return

class CopyStrategy(GSStrategy):
    """Makes a bid equal to the value of the card."""
    name = "copy"
    def __init__(self, game_params):
        return

    def start_game(self):
        return

    def get_bid(self, turn_value):
        return turn_value

    def update_history(self, hist):
        return

class CopyP1Strategy(GSStrategy):
    """Makes a bid equal to one more than the value of the card."""
    name = "copy_plus_1"
    def __init__(self, game_params):
        self.n = game_params['length']
        return

    def start_game(self):
        return

    def get_bid(self, turn_value):
        return (turn_value + 1) % self.n

    def update_history(self, hist):
        return

class AntiPureStrategy(GSStrategy):
    """Creates an internal distribution map for the opponent's gameplay on the last round.
    This strategy assumes there is only 1 other player.
    Initial distribution assumes the opposing player will play exactly what the card is.

    Gameplay is optimal against any pure strategy.
    """
    name = "anti_pure"
    def __init__(self, game_params):
        self.n = game_params['length']
        self.players = game_params['players']
        assert len(self.players) == 2
        self.op_name = [pl for pl in self.players if pl != self.name][0]
        self.op_map = {n:n for n in range(self.n + 1)}

    def start_game(self):
        return

    def update_history(self, hist):
        # update the bids made by the op
        self.op_map[hist['turn_value']] = hist['bids'][self.op_name]

    def get_bid(self, turn_value):
        return (self.op_map[turn_value] + 1) % self.n

class MyStrategy(GSStrategy):
    """Makes a bid equal to the value of the card."""
    name = "legend"
    def __init__(self, game_params):
        return

    def start_game(self):
        return

    def get_bid(self, turn_value):
        if turn_value == 1:
            return 2
        elif turn_value == 2:
            return 1
        return turn_value

    def update_history(self, hist):
        return

class EVThresholdStrategy(GSStrategy):
    """
    Exploits random opponents by bidding minimally
    above expected opponent bids.
    """
    name = "ev_threshold"

    def __init__(self, game_params):
        self.n = game_params['length']

    def start_game(self):
        self.remaining_bids = list(range(self.n))
        self.remaining_prizes = list(range(self.n))

    def get_bid(self, turn_value):
        prizes = sorted(self.remaining_prizes)
        bids = sorted(self.remaining_bids)

        prize_rank = prizes.index(turn_value)
        prize_ratio = prize_rank / len(prizes)

        # Sacrifice low-value prizes
        if prize_ratio < 0.3:
            bid = bids[0]
        else:
            expected_opponent_bid = sum(bids) / len(bids)

            # smallest bid that beats expected opponent bid
            bid = next(
                (b for b in bids if b > expected_opponent_bid),
                bids[-1]
            )

        self.remaining_bids.remove(bid)
        self.remaining_prizes.remove(turn_value)

        return bid

    def update_history(self, hist):
        return

class RankMatchStrategy(GSStrategy):
    """
    Bids according to the rank of the current prize card
    among remaining prize cards.
    """
    name = "rank_match"

    def __init__(self, game_params):
        self.n = game_params['length']

    def start_game(self):
        self.remaining_bids = list(range(self.n))
        self.remaining_prizes = list(range(self.n))

    def get_bid(self, turn_value):
        # sort remaining prizes and bids
        prizes = sorted(self.remaining_prizes)
        bids = sorted(self.remaining_bids)

        # find rank of current prize
        rank = prizes.index(turn_value - 1)

        bid = bids[rank]

        # remove used bid and prize
        self.remaining_bids.remove(bid)
        self.remaining_prizes.remove(turn_value - 1)

        return bid + 1

    def update_history(self, hist):
        return
