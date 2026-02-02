"""Expert Bot implementation for OpenSpiel with conversation history support"""

import os
import pyspiel
import re
import time
import concurrent.futures
import pexpect

from othellobot import DynamicMinimaxAI
from typing import Tuple, Optional, Dict, List, Any

from base_agent import BaseGameAgent

# Constants
DEFAULT_MAX_PARSING_RETRIES = 2



PROMPT_RE = r"\r?\n>\s*$"
MOVE_RE = re.compile(r"\b([a-h][1-8])\b", re.IGNORECASE)

EDAX_BIN  = "/root/workspace/openspiel-bot/edax-reversi/bin/lEdax-x86-64-v3"
EDAX_ROOT = "/root/workspace/openspiel-bot/edax-reversi"


class ParsingError(Exception):
    """Raised when action parsing fails after all retry attempts"""
    pass

class EdaxClient:
    """
    Minimal Edax console driver:
      - set_board(setboard_65)
      - hint(k) -> List[(move, score)]
    """

    def __init__(self, edax_bin: str, edax_root: str, timeout: int = 60):
        self.edax_bin = edax_bin
        self.edax_root = edax_root
        self.timeout = timeout
        self.child: Optional[pexpect.spawn] = None

    def start(self) -> None:
        if self.child is not None and self.child.isalive():
            return
        self.child = pexpect.spawn(
            self.edax_bin,
            cwd=self.edax_root,
            encoding="utf-8",
            timeout=self.timeout,
        )
        self.child.expect(PROMPT_RE)

    def close(self) -> None:
        if self.child is None:
            return
        if self.child.isalive():
            try:
                self._cmd("quit", expect_prompt=False)
            except Exception:
                pass
        try:
            self.child.close()
        finally:
            self.child = None

    def _cmd(self, s: str, expect_prompt: bool = True) -> str:
        if self.child is None:
            raise RuntimeError("EdaxClient is not started. Call start().")
        self.child.sendline(s)
        if not expect_prompt:
            self.child.expect(pexpect.EOF)
            return self.child.before
        self.child.expect(PROMPT_RE)
        return self.child.before

    @staticmethod
    def _parse_hint_move_scores(hint_output: str, k: int) -> List[Tuple[str, int]]:
        """
        Parse Edax 'hint k' table and return [(move, score)] in table order.
        For rows like:
          21@73%  -18  0:00.404  ...  f5 D7 c3 ...
        move = first PV move, score = column 2
        """
        lines = hint_output.splitlines()

        sep_idxs = [i for i, ln in enumerate(lines) if ln.strip().startswith("------+")]
        if len(sep_idxs) < 2:
            return []

        start = sep_idxs[0] + 1
        end = sep_idxs[1]

        results: List[Tuple[str, int]] = []
        for ln in lines[start:end]:
            s = ln.strip()
            if not s:
                continue

            toks = s.split()
            if len(toks) < 6:
                continue

            score_tok = toks[1]
            pv_first = toks[5]

            if not re.fullmatch(r"[+-]?\d+", score_tok):
                continue
            if not MOVE_RE.fullmatch(pv_first.lower()):
                continue

            results.append((pv_first.lower(), int(score_tok)))
            if len(results) >= k:
                break

        return results

    def set_board(self, setboard_65: str) -> None:
        """
        setboard_65: 65-char string = 64 board chars + 1 turn char.
        """
        if len(setboard_65) != 65:
            raise ValueError(f"setboard string must be 65 chars, got {len(setboard_65)}")
        self._cmd(f"setboard {setboard_65}")

    def hint(self, k: int = 6) -> List[Tuple[str, int]]:
        """
        Returns top-k candidate moves and their scores: [(move, score), ...]
        """
        out = self._cmd(f"hint {k}")
        return self._parse_hint_move_scores(out, k)

    def hint_raw(self, k: int = 6) -> str:
        """
        Returns raw hint output (useful for debugging/parsing changes).
        """
        return self._cmd(f"hint {k}")

    def play(self, move: str) -> None:
        """
        move: coordinate like 'd3'
        """
        move = move.strip().lower()
        if not MOVE_RE.fullmatch(move):
            raise ValueError(f"Invalid move format: {move}")
        self._cmd(move)

    # Optional: allow "with EdaxClient(...) as e:"
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()



class OthelloTestBot(pyspiel.Bot):
    """
    Wraps ExpertBot as an OpenSpiel Bot with conversation history management

    This implementation maintains full conversation history and supports 
    retry mechanism with context-aware error feedback.
    """
    def __init__(
        self,
        game: pyspiel.Game,
        player_id: int,
        agent: BaseGameAgent,
        seed: Optional[int] = None,
        max_parsing_retries: int = DEFAULT_MAX_PARSING_RETRIES,
        executor: concurrent.futures.ThreadPoolExecutor = None,
        verbose: Optional[bool] = False,
    ):
        """
        Initialize Expert Bot with conversation history support
        """
        pyspiel.Bot.__init__(self)
        self._game = game
        self._player_id = player_id
        self._seed = seed
        self._agent = agent
        self._executor = executor
        self._max_parsing_retries = max_parsing_retries
        self._othello_bot = DynamicMinimaxAI()
        self.edax = EdaxClient(EDAX_BIN, EDAX_ROOT)
        self.edax.start()

        self._conversation: List[Dict[str, str]] = []
        self._action_history: List[Dict[str, Any]] = []
        self._system_prompt_generated = False
        self._last_error: Optional[str] = None
        self._total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self._observation: Optional[str] = None
        self._verbose = verbose

    def restart_at(self, state):
        """Reset to new game"""
        self._conversation.clear()
        self._action_history.clear()
        self._system_prompt_generated = False
        self._last_error = None
        self._total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self._observation = None

    def inform_action(self, state, player_id, action):
        """Record all players' actions for game replay and verification"""
        try:
            action_str = state.action_to_string(player_id, action)
        except:
            action_str = str(action)
        
        # Convert numpy types to Python native types for JSON serialization
        self._action_history.append({
            "player_id": int(player_id),
            "action": int(action),
            "action_str": action_str,
            "is_llm": bool(player_id == self._player_id)
        })

        try:
            self._observation = state.observation_string()
        except:
            try:
                self._observation = str(state)
            except:
                self._observation = None

    def step(self, state):
        """
        Core method: choose action with conversation history and retry mechanism

        This is called by evaluate_bots during game play.
        """
        step_start_time = time.time()
        # Generate system prompt (first time only)
        if not self._system_prompt_generated:
            system_prompt = self._agent.generate_system_prompt()
            self._conversation.append({"role": "system", "content": system_prompt})
            self._system_prompt_generated = True

        # Get legal actions ONCE at the start of this turn
        legal_actions = state.legal_actions(self._player_id)

        # Generate user prompt
        user_prompt = self._agent.generate_user_prompt(
            state=state,
            player_id=self._player_id,
            legal_actions=legal_actions
        )
        self._conversation.append({"role": "user", "content": user_prompt})

        def copy(board):
            """
            盤面をコピーする関数。
            board: 2次元配列のオセロボード
            """
            return [row[:] for row in board]

        # Retry loop for parsing
        if self._verbose:
            print(f"Since step function: {(time.time() - step_start_time) / 1000}s")
        for attempt in range(self._max_parsing_retries + 1):
            # response = 
            observation = state.observation_string()
            observation = observation.split("\n")
            display = ""
            board = []
            for i in range(8):
                _ = []
                s = observation[i + 2][1:-1]
                for c in s.split(" "):
                    if c == "-":
                        _.append(0)
                    elif c == 'o':
                        _.append(1)
                    elif c == 'x':
                        _.append(2)
                    display += c + " "
                display += "\n"
                board.append(_)

            observation = state.observation_string()
            observation = observation.split("\n")
            setboard_str = ""
            display = ""
            for i in range(8):
                s = observation[i + 2][1:-1]
                _s = ""
                for c in s:
                    if c != " ":
                        _s += c
                setboard_str += _s
                _s += "\n"
                display += _s

            setboard_str += observation[0][7]
            self.edax.set_board(setboard_str)

            print(display, flush=True)
            top = self.edax.hint(20)
            if self._verbose:
                print(state.observation_string())
                print("TOP-K (move, score):", top)
                print(state.legal_actions(self._player_id))

            # eval = self._evaluator.evaluate(board, BLACK)
            # print(f"eval = {eval}", flush=True)

            x, y = self._othello_bot.place(copy(board), 2 - self._player_id)
            x, y = x + 1, y + 1
            action_str = f"{chr(x-1+ord('a'))}{chr(y-1+ord('1'))}"
            if self._verbose:
                print(observation, flush=True)
                print(action_str, flush=True)
                print(f"x = {x}, y = {y}", flush=True)
                print(state.legal_actions(self._player_id), flush=True)
            cnt = 0
            action_id = -1
            legal_action = state.legal_actions(self._player_id)

            for action in legal_actions:
                action_id_str = state.action_to_string(self._player_id, action)
                if action_id_str == action_str:
                    action_id = action
                    break

            if action_id == -1 or action_id not in legal_action:
                if self._verbose:
                    print("Nothing is selected by Expert bot!")
                action_id = legal_action[0]

            response = str(action_id)

            self._conversation.append({"role": "assistant", "content": response})

            result = self._parse_action(response, state, legal_actions)

            if result['success']:
                # Success: record action and return
                action = result['action']
                self.inform_action(state, self._player_id, action)
                if self._verbose:
                    print(f"Action took: {(time.time() - step_start_time) / 1000}s")
                return action

            # Parsing failed - use simplified error message to avoid response contamination
            error_msg = (
                f"Invalid response format. "
                f"You must respond with ONLY the action ID number (e.g., '5'). "
                f"This is attempt {attempt + 1} of {self._max_parsing_retries + 1}."
            )
            self._conversation.append({"role": "user", "content": error_msg})
            if attempt >= self._max_parsing_retries:
                raise ParsingError(
                    f"Failed to parse valid action after {self._max_parsing_retries + 1} retries. "
                    f"Last response: '{response}'. Error: {result['error_message']}"
                )
        
        raise RuntimeError("Should not reach here")


    def _parse_action(self, response: str, state, legal_actions: List[int]) -> Dict:
        """
        Robust action parsing with multiple strategies
        
        Returns dict with keys: success, action, error_message, matched_method
        """
        response_clean = response.strip()
        
        # Strategy 1: Pure number (highest priority)
        if match := re.search(r'^\s*(\d+)\s*$', response_clean):
            try:
                action = int(match.group(1))
                if action in legal_actions:
                    return {'success': True, 'action': action, 'error_message': '', 'matched_method': 'pure_number'}
                else:
                    return {
                        'success': False,
                        'action': None,
                        'error_message': f"Number {action} not in legal actions: {legal_actions}",
                        'matched_method': 'number_invalid'
                    }
            except ValueError as e:
                return {
                    'success': False,
                    'action': None,
                    'error_message': f"Cannot convert to integer: {str(e)}. Model generated invalid action.",
                    'matched_method': 'number_conversion_error'
                }
        
        # Strategy 2: Find legal action ID in text
        for action in legal_actions:
            if re.search(rf'\b{action}\b', response_clean):
                return {'success': True, 'action': action, 'error_message': '', 'matched_method': 'number_in_text'}
        
        # Strategy 3: Match action string (exact or simplified)
        action_map = self._build_action_string_map(state, legal_actions)
        response_lower = response_clean.lower()
        response_simplified = re.sub(r'[^a-z0-9]', '', response_lower)
        
        # Try exact match first, then simplified
        for action_str, action_id in action_map.items():
            if action_str in response_lower:
                return {'success': True, 'action': action_id, 'error_message': '', 'matched_method': 'string_exact'}
            simplified = re.sub(r'[^a-z0-9]', '', action_str)
            if simplified and simplified in response_simplified:
                return {'success': True, 'action': action_id, 'error_message': '', 'matched_method': 'string_simplified'}
        
        return {
            'success': False,
            'action': None,
            'error_message': f"Cannot parse action from: '{response_clean}'. Expected format: just the action ID number (e.g., '5').",
            'matched_method': 'failed'
        }


    def get_conversation(self):
        """Get conversation history (for debugging)"""
        return self._conversation
    
    def get_action_history(self):
        """Get complete action history for all players"""
        return self._action_history

    def get_last_error(self):
        """Get last error string (if any)"""
        return self._last_error

    def get_total_usage(self):
        """Get accumulated usage statistics"""
        return self._total_usage
    
    def get_observation(self):
        """Get final observation string"""
        return self._observation

