"""Expert Bot implementation for OpenSpiel with conversation history support"""

import os
import pyspiel
import re
import time
import concurrent.futures
from typing import Tuple, Optional, Dict, List, Any

from base_agent import BaseGameAgent

# Constants
DEFAULT_MAX_PARSING_RETRIES = 2




class ParsingError(Exception):
    """Raised when action parsing fails after all retry attempts"""
    pass


class LiarsDiceManualBot(pyspiel.Bot):
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
        verbose: bool = False,
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

        # Retry loop for parsing
        if self._verbose:
            print(f"Since step function: {(time.time() - step_start_time) / 1000}s")
        for attempt in range(self._max_parsing_retries + 1):
            try:
                observation = state.observation_string()
            except:
                try:
                    observation = str(state)
                except:
                    observation = None

            print(observation)
            legal_actions = state.legal_actions(self._player_id)
            print("Legal actions:")
            for action_id in legal_actions:
                try:
                    action_str = state.action_to_string(self._player_id, action_id)
                    print(f"  {action_id} -> {action_str}")
                except:
                    print(f"  {action_id} -> (action {action_id})")

            while True:
                action_id = input("Enter action ID: ")
                try:
                    action_id = int(action_id)
                except ValueError:
                    print("Invalid action ID, try again")
                    continue

                if action_id not in state.legal_actions(self._player_id):
                    print("Action ID not in legal actions, try again")
                else:
                    break


            response = str(action_id)
            # print(num_cards, current_reward, goofspiel_strategy.get_bid(current_reward), response)
            # print(legal_actions)

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
