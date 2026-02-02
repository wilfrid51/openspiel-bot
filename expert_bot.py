"""Expert Bot implementation for OpenSpiel with conversation history support"""

import os
from sys import implementation
import pyspiel
import numpy as np
import asyncio
import re
import time
import random
import concurrent.futures
import pexpect
from typing import Tuple, Optional, Dict, List, Any

from base_agent import BaseGameAgent

from goofspiel.strategies import *

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


class HexGame:
    """Main Hex game logic class"""
    
    MAX_SIZE = 13
    
    def __init__(self, size: int = 11):
        """Initialize a Hex game with given board size"""
        self.size = size
        self.max_fld = size * size
        
        # Board state: -1 = red, 0 = empty, 1 = blue
        self.fld = [[0 for _ in range(self.MAX_SIZE)] for _ in range(self.MAX_SIZE)]
        
        # Potential values for each position (4 directions)
        self.pot = [[[0 for _ in range(4)] for _ in range(self.MAX_SIZE)] 
                    for _ in range(self.MAX_SIZE)]
        
        # Bridge values for strategic positions
        self.bridge = [[[0 for _ in range(4)] for _ in range(self.MAX_SIZE)] 
                       for _ in range(self.MAX_SIZE)]
        
        # Update flags for potential calculation
        self.upd = [[True for _ in range(self.MAX_SIZE)] for _ in range(self.MAX_SIZE)]
        
        # Move history
        self.history = [[0, 0] for _ in range(self.max_fld + 1)]
        self.move_count = 0
        self.max_move_count = 0
        
        # Game settings
        self.is_start_0 = True  # Does player 0 (red) start?
        self.start_0 = True
        self.is_swap = False
        self.is_over = True
        self.active_color = 0
        
        # Player settings: True = human, False = AI
        self.is_player = [True, False]
        self.level = [2, 3]  # AI difficulty levels
        
    def init_game(self):
        """Initialize/reset the game board"""
        # Clear board
        for i in range(self.size):
            for j in range(self.size):
                self.fld[i][j] = 0
        
        self.start_0 = self.is_start_0
        self.move_count = 0
        self.max_move_count = 0
        self.is_over = False

    def set_board(self, i: int, j: int, value: int):
        """Set the board value at position (i, j)"""
        self.fld[i - 1][j - 1] = value
    
    def sync_move_count(self):
        """Update move_count based on current board state"""
        self.move_count = 0
        for i in range(self.size):
            for j in range(self.size):
                if self.fld[i][j] != 0:
                    self.move_count += 1

    def make_move(self, ii: int, jj: int, check_win: bool = False) -> Optional[int]:
        """
        Make a move at position (ii, jj)
        
        Args:
            ii, jj: Board coordinates
            check_win: Whether to check for win condition
            
        Returns:
            Winner color if game is over, None otherwise
        """
        iis, jjs = ii, jj
        
        # Handle swap rule on move 1
        if self.move_count == 1:
            if self.fld[ii][jj] != 0:
                self.fld[ii][jj] = 0
                iis, jjs = jj, ii
                self.is_swap = True
            else:
                self.is_swap = False
        
        # Place stone
        ccol = ((self.move_count + 1 + self.start_0) % 2) * 2 - 1
        self.fld[iis][jjs] = ccol
        
        # Update history
        if self.history[self.move_count][0] != ii:
            self.history[self.move_count][0] = ii
            self.max_move_count = self.move_count + 1
        if self.history[self.move_count][1] != jj:
            self.history[self.move_count][1] = jj
            self.max_move_count = self.move_count + 1
        
        self.move_count += 1
        if self.max_move_count < self.move_count:
            self.max_move_count = self.move_count
        
        # Check win condition if requested
        if check_win:
            self.get_pot(0)
            
            if ccol < 0:  # Red
                if self.pot[ii][jj][2] > 0 or self.pot[ii][jj][3] > 0:
                    return None
                self.is_over = True
                return -1  # Red wins
            else:  # Blue
                if self.pot[ii][jj][0] > 0 or self.pot[ii][jj][1] > 0:
                    return None
                self.is_over = True
                return 1  # Blue wins
        
        return None
    
    def get_pot(self, llevel: int):
        """
        Calculate potential values for all positions
        
        This is the core AI evaluation function that calculates how well
        connected each empty cell is to each border.
        
        Args:
            llevel: Calculation depth level
        """
        dd = 128  # Base distance value
        
        self.active_color = ((self.move_count + 1 + self.start_0) % 2) * 2 - 1
        
        # Initialize all potentials to high value
        for i in range(self.size):
            for j in range(self.size):
                for k in range(4):
                    self.pot[i][j][k] = 20000
                    self.bridge[i][j][k] = 0
        
        # Set border potentials
        # Blue borders (left and right, directions 0 and 1)
        for i in range(self.size):
            if self.fld[i][0] == 0:
                self.pot[i][0][0] = dd  # Left border
            elif self.fld[i][0] > 0:
                self.pot[i][0][0] = 0
            
            if self.fld[i][self.size - 1] == 0:
                self.pot[i][self.size - 1][1] = dd  # Right border
            elif self.fld[i][self.size - 1] > 0:
                self.pot[i][self.size - 1][1] = 0
        
        # Red borders (top and bottom, directions 2 and 3)
        for j in range(self.size):
            if self.fld[0][j] == 0:
                self.pot[0][j][2] = dd  # Top border
            elif self.fld[0][j] < 0:
                self.pot[0][j][2] = 0
            
            if self.fld[self.size - 1][j] == 0:
                self.pot[self.size - 1][j][3] = dd  # Bottom border
            elif self.fld[self.size - 1][j] < 0:
                self.pot[self.size - 1][j][3] = 0
        
        # Calculate blue potential (directions 0 and 1)
        for kk in range(2):
            for i in range(self.size):
                for j in range(self.size):
                    self.upd[i][j] = True
            
            nn = 0
            while nn < 12:
                nn += 1
                bb = 0
                
                # Forward pass
                for i in range(self.size):
                    for j in range(self.size):
                        if self.upd[i][j]:
                            bb += self.set_pot(i, j, kk, 1, llevel)
                
                # Backward pass
                for i in range(self.size - 1, -1, -1):
                    for j in range(self.size - 1, -1, -1):
                        if self.upd[i][j]:
                            bb += self.set_pot(i, j, kk, 1, llevel)
                
                if bb == 0:
                    break
        
        # Calculate red potential (directions 2 and 3)
        for kk in range(2, 4):
            for i in range(self.size):
                for j in range(self.size):
                    self.upd[i][j] = True
            
            nn = 0
            while nn < 12:
                nn += 1
                bb = 0
                
                # Forward pass
                for i in range(self.size):
                    for j in range(self.size):
                        if self.upd[i][j]:
                            bb += self.set_pot(i, j, kk, -1, llevel)
                
                # Backward pass
                for i in range(self.size - 1, -1, -1):
                    for j in range(self.size - 1, -1, -1):
                        if self.upd[i][j]:
                            bb += self.set_pot(i, j, kk, -1, llevel)
                
                if bb == 0:
                    break
    
    def set_pot(self, ii: int, jj: int, kk: int, cc: int, llevel: int) -> int:
        """
        Set potential value for a single cell
        
        Args:
            ii, jj: Cell coordinates
            kk: Direction (0-3)
            cc: Color (1 for blue, -1 for red)
            llevel: Calculation level
            
        Returns:
            1 if potential was updated, 0 otherwise
        """
        self.upd[ii][jj] = False
        self.bridge[ii][jj][kk] = 0
        
        if self.fld[ii][jj] == -cc:
            return 0
        
        dd = 140  # Empty cell cost
        bb = 66 if cc == self.active_color else 52  # Bridge bonus
        
        # Get potential values of 6 neighbors
        vv = [0] * 6
        tt = [0] * 6
        
        vv[0] = self.pot_val(ii + 1, jj, kk, cc)
        vv[1] = self.pot_val(ii, jj + 1, kk, cc)
        vv[2] = self.pot_val(ii - 1, jj + 1, kk, cc)
        vv[3] = self.pot_val(ii - 1, jj, kk, cc)
        vv[4] = self.pot_val(ii, jj - 1, kk, cc)
        vv[5] = self.pot_val(ii + 1, jj - 1, kk, cc)
        
        ddb = 0
        
        # Check for bridge patterns
        for ll in range(6):
            # Adjacent stones with enemy between
            if vv[ll] >= 30000 and vv[(ll + 2) % 6] >= 30000:
                if vv[(ll + 1) % 6] < 0:
                    ddb += 32
                else:
                    vv[(ll + 1) % 6] += 128
        
        # Opposite stones bonus
        for ll in range(6):
            if vv[ll] >= 30000 and vv[(ll + 3) % 6] >= 30000:
                ddb += 30
        
        # Find minimum and count
        mm = 30000
        for ll in range(6):
            if vv[ll] < 0:
                vv[ll] += 30000
                tt[ll] = 10
            else:
                tt[ll] = 1
            if mm > vv[ll]:
                mm = vv[ll]
        
        nn = sum(tt[ll] for ll in range(6) if vv[ll] == mm)
        
        # Calculate bridge value
        if llevel > 1:
            self.bridge[ii][jj][kk] = nn // 5
            
            if 2 <= nn < 10:
                self.bridge[ii][jj][kk] = bb + nn - 2
                mm -= 32
            
            if nn < 2:
                oo = 30000
                for ll in range(6):
                    if vv[ll] > mm and oo > vv[ll]:
                        oo = vv[ll]
                
                if oo <= mm + 104:
                    self.bridge[ii][jj][kk] = bb - (oo - mm) // 4
                    mm -= 64
                
                mm = (mm + oo) // 2
        
        # Adjust bridge value based on position
        if 0 < ii < self.size - 1 and 0 < jj < self.size - 1:
            self.bridge[ii][jj][kk] += ddb
        else:
            self.bridge[ii][jj][kk] -= 2
        
        if ((ii == 0 or ii == self.size - 1) and 
            (jj == 0 or jj == self.size - 1)):
            self.bridge[ii][jj][kk] //= 2
        
        if self.bridge[ii][jj][kk] > 68:
            self.bridge[ii][jj][kk] = 68
        
        # Update potential
        if self.fld[ii][jj] == cc:
            if mm < self.pot[ii][jj][kk]:
                self.pot[ii][jj][kk] = mm
                self._set_neighbors_upd(ii, jj, cc)
                return 1
            return 0
        
        if mm + dd < self.pot[ii][jj][kk]:
            self.pot[ii][jj][kk] = mm + dd
            self._set_neighbors_upd(ii, jj, cc)
            return 1
        
        return 0
    
    def pot_val(self, ii: int, jj: int, kk: int, cc: int) -> int:
        """Get potential value at position, handling boundaries"""
        if ii < 0 or jj < 0 or ii >= self.size or jj >= self.size:
            return 30000
        
        if self.fld[ii][jj] == 0:
            return self.pot[ii][jj][kk]
        
        if self.fld[ii][jj] == -cc:
            return 30000
        
        return self.pot[ii][jj][kk] - 30000
    
    def _set_neighbors_upd(self, ii: int, jj: int, cc: int):
        """Mark neighbors for update"""
        neighbors = [
            (ii + 1, jj), (ii, jj + 1), (ii - 1, jj + 1),
            (ii - 1, jj), (ii, jj - 1), (ii + 1, jj - 1)
        ]
        
        for ni, nj in neighbors:
            if 0 <= ni < self.size and 0 <= nj < self.size:
                self.upd[ni][nj] = True
    
    def get_best_move(self, the_col: int, the_level: int) -> Tuple[int, int]:
        """
        AI move selection using potential-based evaluation
        
        Args:
            the_col: Color to move (-1 for red, 1 for blue)
            the_level: AI difficulty level
            
        Returns:
            (ii, jj): Best move coordinates
        """
        vv = {}
        ff = 0
        if self.move_count > 0:
            ff = 190 / (self.move_count * self.move_count)
        
        mm = 20000
        ii_b, jj_b = 0, 0
        
        # Calculate center of mass for occupied positions
        center = self.size // 2
        ii_q, jj_q = 0, 0
        for i in range(self.size):
            for j in range(self.size):
                if self.fld[i][j] != 0:
                    ii_q += 2 * i + 1 - self.size
                    jj_q += 2 * j + 1 - self.size

        ii_q = self._sign(ii_q)
        jj_q = self._sign(jj_q)
        
        # Evaluate all empty positions
        for i in range(self.size):
            for j in range(self.size):
                if self.fld[i][j] == 0:
                    # Random component (decreases with difficulty)
                    mmp = random.random() * max(0, 49 - the_level * 16)
                    
                    # Distance from center penalty (decreases over time)
                    mmp += (abs(i - center) + abs(j - center)) * ff
                    
                    # Tendency toward center of mass
                    mmp += 8 * (ii_q * (i - center) + jj_q * (j - center)) / (self.move_count + 1)
                    
                    # Bridge values (higher levels consider this)
                    if the_level > 2:
                        for k in range(4):
                            mmp -= self.bridge[i][j][k]
                    
                    # Potential values (connectivity to borders)
                    pp0 = self.pot[i][j][0] + self.pot[i][j][1]  # Blue
                    pp1 = self.pot[i][j][2] + self.pot[i][j][3]  # Red
                    mmp += pp0 + pp1
                    
                    # Strong penalty if not connected to both borders
                    if pp0 <= 268 or pp1 <= 268:
                        mmp -= 400
                    
                    vv[i * self.size + j] = mmp
                    
                    if mmp < mm:
                        mm = mmp
                        ii_b, jj_b = i, j
        
        # Advanced tactical checks (level > 2)
        if the_level > 2:
            mm += 108
            ii_b, jj_b = self._check_tactical_moves(
                vv, mm, the_col, ii_b, jj_b
            )
        
        return ii_b, jj_b
    
    def _check_tactical_moves(self, vv: dict, mm: float, 
                             the_col: int, ii_b: int, jj_b: int) -> Tuple[int, int]:
        """Check for tactical defensive moves"""
        for i in range(self.size):
            for j in range(self.size):
                key = i * self.size + j
                if key not in vv or vv[key] >= mm:
                    continue
                
                if the_col < 0:  # Red
                    # Check various defensive patterns
                    if 3 < i < self.size - 1 and 0 < j < 3:
                        if self.fld[i - 1][j + 2] == -the_col:
                            cc = self.can_connect_far_border(
                                i - 1, j + 2, -the_col
                            )
                            if cc < 2:
                                ii_temp = i
                                if cc < -1:
                                    ii_temp -= 1
                                    cc += 1
                                jj_temp = j - cc
                                mm = vv[key]
                                ii_b, jj_b = ii_temp, jj_temp
                    
                    # Additional red patterns...
                    if 0 < i < self.size - 1 and j == 0:
                        if (self.fld[i - 1][j + 2] == -the_col and
                            self.fld[i - 1][j] == 0 and
                            self.fld[i - 1][j + 1] == 0 and
                            self.fld[i][j + 1] == 0 and
                            self.fld[i + 1][j] == 0):
                            ii_b, jj_b = i, j
                            mm = vv[key]
                
                else:  # Blue
                    # Check various defensive patterns for blue
                    if 3 < j < self.size - 1 and 0 < i < 3:
                        if self.fld[i + 2][j - 1] == -the_col:
                            cc = self.can_connect_far_border(
                                i + 2, j - 1, -the_col
                            )
                            if cc < 2:
                                jj_temp = j
                                if cc < -1:
                                    jj_temp -= 1
                                    cc += 1
                                ii_temp = i - cc
                                mm = vv[key]
                                ii_b, jj_b = ii_temp, jj_temp
                    
                    # Additional blue patterns...
                    if 0 < j < self.size - 1 and i == 0:
                        if (self.fld[i + 2][j - 1] == -the_col and
                            self.fld[i][j - 1] == 0 and
                            self.fld[i + 1][j - 1] == 0 and
                            self.fld[i + 1][j] == 0 and
                            self.fld[i][j + 1] == 0):
                            ii_b, jj_b = i, j
                            mm = vv[key]
        
        return ii_b, jj_b
    
    def can_connect_far_border(self, nn: int, mm: int, cc: int) -> int:
        """
        Check if a position can connect to far border without interference
        
        Returns:
            0-1: Can connect easily
            2: Cannot connect (blocked)
            -1, -2: Can connect with adjustment
        """
        if cc > 0:  # Blue
            if 2 * mm < self.size - 1:
                # Check near border
                for i in range(self.size):
                    for j in range(mm):
                        if (j - i < mm - nn and 
                            i + j <= nn + mm and 
                            self.fld[i][j] != 0):
                            return 2
                
                if self.fld[nn - 1][mm] == -cc:
                    return 0
                if self.fld[nn - 1][mm - 1] == -cc:
                    if self._get_fld(nn + 2, mm - 1) == -cc:
                        return 0
                    return -1
                if self._get_fld(nn + 2, mm - 1) == -cc:
                    return -2
            else:
                # Check far border
                for i in range(self.size):
                    for j in range(mm + 1, self.size):
                        if (j - i > mm - nn and 
                            i + j >= nn + mm and 
                            self.fld[i][j] != 0):
                            return 2
                
                if self.fld[nn + 1][mm] == -cc:
                    return 0
                if self.fld[nn + 1][mm + 1] == -cc:
                    if self._get_fld(nn - 2, mm + 1) == -cc:
                        return 0
                    return -1
                if self._get_fld(nn - 2, mm + 1) == -cc:
                    return -2
        
        else:  # Red (similar logic for vertical)
            if 2 * nn < self.size - 1:
                for j in range(self.size):
                    for i in range(nn):
                        if (i - j < nn - mm and 
                            i + j <= nn + mm and 
                            self.fld[i][j] != 0):
                            return 2
                
                if self.fld[nn][mm - 1] == -cc:
                    return 0
                if self.fld[nn - 1][mm - 1] == -cc:
                    if self._get_fld(nn - 1, mm + 2) == -cc:
                        return 0
                    return -1
                if self._get_fld(nn - 1, mm + 2) == -cc:
                    return -2
            else:
                for j in range(self.size):
                    for i in range(nn + 1, self.size):
                        if (i - j > nn - mm and 
                            i + j >= nn + mm and 
                            self.fld[i][j] != 0):
                            return 2
                
                if self.fld[nn][mm + 1] == -cc:
                    return 0
                if self.fld[nn + 1][mm + 1] == -cc:
                    if self._get_fld(nn + 1, mm - 2) == -cc:
                        return 0
                    return -1
                if self._get_fld(nn + 1, mm - 2) == -cc:
                    return -2
        
        return 1
    
    def _get_fld(self, ii: int, jj: int) -> int:
        """Get field value with boundary handling"""
        if ii < 0:
            return -1
        if jj < 0:
            return 1
        if ii >= self.size:
            return -1
        if jj >= self.size:
            return 1
        return self.fld[ii][jj]
    
    @staticmethod
    def _sign(xx: int) -> int:
        """Sign function"""
        if xx < 0:
            return -1
        if xx > 0:
            return 1
        return 0
    
    def back(self):
        """Undo last move"""
        if self.move_count > 0:
            self.is_over = False
            self.move_count -= 1
            
            ii = self.history[self.move_count][0]
            jj = self.history[self.move_count][1]
            
            if self.move_count == 1 and self.is_swap:
                self.fld[jj][ii] = 0
                self.fld[ii][jj] = ((self.move_count + self.start_0) % 2) * 2 - 1
            else:
                self.fld[ii][jj] = 0
    
    def get_move_list(self) -> str:
        """Get list of moves in algebraic notation (e.g., 'A1 B2 C3')"""
        moves = []
        for n in range(self.max_move_count):
            i = self.history[n][0]
            j = self.history[n][1]
            moves.append(f"{chr(65 + j)}{i + 1}")
        return " ".join(moves)
    
    def apply_move_list(self, move_list: str) -> bool:
        """
        Apply a sequence of moves from algebraic notation
        
        Args:
            move_list: String like "A1 B2 C3"
            
        Returns:
            True if successful, False if invalid
        """
        self.init_game()
        moves = move_list.strip().split()
        
        for n, move in enumerate(moves):
            if len(move) < 2:
                return False
            
            j = ord(move[0]) - 65
            try:
                i = int(move[1:]) - 1
            except ValueError:
                return False
            
            if i < 0 or j < 0 or i >= self.size or j >= self.size:
                return False
            
            check_win = (n == len(moves) - 1)
            self.make_move(i, j, check_win)
        
        return True
    
    def print_board(self):
        """Print current board state to console"""
        print("\n  ", end="")
        for j in range(self.size):
            print(f" {chr(65 + j)}", end="")
        print()
        
        for i in range(self.size):
            print(f"{i + 1:2d}", end=" ")
            print(" " * i, end="")
            
            for j in range(self.size):
                if self.fld[i][j] == -1:
                    print(" R", end="")
                elif self.fld[i][j] == 1:
                    print(" B", end="")
                else:
                    print(" .", end="")
            print()
        print()

    def print_values(self, ii):
        """Print current board state to console"""
        print("\n     ", end="")
        for j in range(self.size):
            print(f" {chr(65 + j):5s}", end="")
        print()
        
        for i in range(self.size):
            print(f"{i + 1:5d}", end=" ")
            print(" " * i, end="")
            
            for j in range(self.size):
                print(f"{self.pot[i][j][ii]:5d} ", end="")
            print()
        print()
    
    def get_winner(self) -> Optional[int]:
        """
        Check if game is won
        
        Returns:
            -1 if red wins, 1 if blue wins, None if no winner yet
        """
        if not self.is_over:
            return None
        
        self.get_pot(0)
        
        # Check if any blue stone connects both borders
        for i in range(self.size):
            for j in range(self.size):
                if self.fld[i][j] == 1:
                    if self.pot[i][j][0] <= 0 and self.pot[i][j][1] <= 0:
                        return 1
        
        # Check if any red stone connects both borders
        for i in range(self.size):
            for j in range(self.size):
                if self.fld[i][j] == -1:
                    if self.pot[i][j][2] <= 0 and self.pot[i][j][3] <= 0:
                        return -1
        
        return None


class OthelloExpertBot(pyspiel.Bot):
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

        # Retry loop for parsing
        if self._verbose:
            print(f"Since step function: {(time.time() - step_start_time) / 1000}s")
        for attempt in range(self._max_parsing_retries + 1):
            # response = 
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

            top = self.edax.hint(6)
            if self._verbose:
                print(state.observation_string())
                print("TOP-K (move, score):", top)
                print(state.legal_actions(self._player_id))
            cnt = 0
            action_id = -1
            legal_action = state.legal_actions(self._player_id)

            while cnt < len(top):
                best = str(top[cnt][0])
                action_id = (ord(best[0]) - ord('a')) + (ord(best[1]) - ord('1')) * 8
                if self._verbose:
                    print(best, action_id)
                cnt += 1
                if action_id in legal_action:
                    break

            if action_id == -1 or action_id not in legal_action:
                if self._verbose:
                    print("Nothing is selected by Expert bot!")
                action_id = legal_action[0]

            if self._verbose:
                print(display, flush=True)

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


class GoofSpielExpertBot(pyspiel.Bot):
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

            _p0_hand = observation.split("\n")[0].split(":")[1].split(" ")
            p0_hand = []
            for c in _p0_hand:
                if c != "":
                    p0_hand.append(int(c))
            _point_card_sequence = observation.split("\n")[2].split(":")[1].split(" ")
            point_card_sequence = []
            for c in _point_card_sequence:
                if c != "":
                    point_card_sequence.append(int(c))
            current_reward = point_card_sequence[-1]
            num_cards = len(p0_hand) + len(point_card_sequence) - 1



            game_params={
                'length': num_cards,
                'players': ["mcts_bot", "mybot"]
            }
            # goofspiel_strategy = RandomStrategy(game_params={'length':num_cards})
            goofspiel_strategy = CopyStrategy(game_params={})
            # goofspiel_strategy = CopyP1Strategy(game_params={'length':num_cards})
            # goofspiel_strategy = AntiPureStrategy(game_params=game_params)
            # goofspiel_strategy = MyStrategy(game_params=game_params)
            # if num_cards > 13:
            #     goofspiel_strategy = CopyStrategy(game_params={})
            # elif num_cards > 15:
            #     goofspiel_strategy = MyStrategy(game_params={'length':num_cards})
            # else:
            #     goofspiel_strategy = RankMatchStrategy(game_params=game_params)

            goofspiel_strategy.start_game()

            action_id = (goofspiel_strategy.get_bid(current_reward) + num_cards - 1) % num_cards

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


class HexExpertBot(pyspiel.Bot):
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

        self._hex_game = HexGame(size=11)

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

            if observation:
                try:
                    # Parse observation to determine board size and state
                    lines = [line for line in observation.split("\n") if line.strip()]

                    # Filter out header lines and find actual board lines
                    board_lines = []
                    for line in lines:
                        # Look for lines with hex board characters
                        stripped = line.strip()
                        if any(c in stripped for c in ['x', 'o', '.']):
                            # Count actual board cells (x, o, .)
                            cell_count = sum(1 for c in stripped if c in 'xo.')
                            if cell_count >= 5:  # At least 5x5 board
                                board_lines.append(stripped)
                    
                    if board_lines:
                        # Determine board size from actual board content
                        # Count cells in the first line
                        first_line = board_lines[0]
                        size = sum(1 for c in first_line if c in 'xo.')

                        # Ensure size is valid (5, 7, 9, or 11)
                        if size not in [5, 7, 9, 11]:
                            # Try to use number of lines
                            size = len(board_lines)
                            if size not in [5, 7, 9, 11]:
                                size = 11  # Default fallback
                        
                        # In OpenSpiel Hex: player 0 uses 'x' (red), player 1 uses 'o' (blue)
                        
                        # Initialize game
                        self._hex_game = HexGame(size=size)
                        self._hex_game.init_game()
                        
                        # Parse board state directly from observation for visualization
                        # This is the board state we'll send to HTML - keep it exactly as parsed
                        board_state_for_viz = []
                        for i, line in enumerate(board_lines):
                            if i >= size:
                                break
                            # Extract cells from line (x, o, or .)
                            cells = [c for c in line if c in 'xo.']
                            row = []
                            for j, cell in enumerate(cells):
                                if j >= size:
                                    break
                                row.append(cell if cell in 'xo' else '.')
                            # Pad row if needed
                            while len(row) < size:
                                row.append('.')
                            board_state_for_viz.append(row[:size])
                        
                        # Also parse for HexGame (for move calculation)
                        # In OpenSpiel Hex, player 0 (x) is typically red (-1), player 1 (o) is blue (1)
                        # But we need to check based on current player
                        for i, line in enumerate(board_lines):
                            if i >= size:
                                break
                            cells = [c for c in line if c in 'xo.']
                            for j, cell in enumerate(cells):
                                if j >= size:
                                    break
                                if cell == 'x':
                                    self._hex_game.fld[i][j] = -1  # x is player 0 (red)
                                elif cell == 'o':
                                    self._hex_game.fld[i][j] = 1   # o is player 1 (blue)
                                else:
                                    self._hex_game.fld[i][j] = 0

                        self._hex_game.print_board()
                        
                        # Sync move count with board state
                        self._hex_game.sync_move_count()
                        
                        # Determine whose turn it is from OpenSpiel state
                        current_player = state.current_player()
                        next_to_move = -1 if current_player == 0 else 1

                        # Calculate best move
                        self._hex_game.get_pot(3)
                        ri, rj = self._hex_game.get_best_move(next_to_move, 3)
                        response = f"{chr(97 + rj)}{ri + 1}"
                        print(response)
                        
                        # Update real-time visualization server if available
                        try:
                            import urllib.request
                            import json
                            # Use the board state directly from observation (not from HexGame)
                            board_state = board_state_for_viz
                            
                            # Send update to visualization server
                            try:
                                # Determine next_to_move for visualization (x or o)
                                next_to_move_char = 'x' if current_player == 0 else 'o'
                                data = json.dumps({
                                    'board': board_state,
                                    'size': size,
                                    'next_move': response,
                                    'next_to_move': next_to_move_char
                                }).encode('utf-8')
                                req = urllib.request.Request(
                                    'http://localhost:8001/update',
                                    data=data,
                                    headers={'Content-Type': 'application/json'}
                                )
                                urllib.request.urlopen(req, timeout=0.1)
                            except:
                                pass  # Server might not be running, that's OK
                        except Exception:
                            pass  # urllib not available or other error
                        
                        # Find matching legal action
                        legal_actions = state.legal_actions(self._player_id)
                        for action in legal_actions:
                            action_str = state.action_to_string(self._player_id, action)
                            if action_str == response:
                                response = str(action)
                                if self._verbose:
                                    print(f"Hex move: {response} ({action_str})")
                                break
                        else:
                            # If no match found, use first legal action as fallback
                            if legal_actions:
                                response = str(legal_actions[0])
                            else:
                                response = None
                    else:
                        # Fallback if parsing fails
                        legal_actions = state.legal_actions(self._player_id)
                        response = str(legal_actions[0]) if legal_actions else None
                except Exception as e:
                    # Fallback on any error
                    if self._verbose:
                        print(f"Error parsing Hex board: {e}")
                    legal_actions = state.legal_actions(self._player_id)
                    response = str(legal_actions[0]) if legal_actions else None
            else:
                # Fallback if no observation
                legal_actions = state.legal_actions(self._player_id)
                response = str(legal_actions[0]) if legal_actions else None
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
