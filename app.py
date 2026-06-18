"""
Streamlit Web Application — Tic-Tac-Toe vs RL Agent (Strength-Optimized).
===========================================================================
Modes:
  • Human vs AI
  • AI vs Random  (watch mode)
  • AI vs Minimax (watch mode)

All bugs fixed:
  1. Zero-sum Bellman negation in _single_q_update and _double_q_update.
  2. last_transition block guarded to opponent-only termination.
  3. Evaluation always uses first_player=1.
  4. MinimaxAgent cache key now includes max_player (was causing X/O cache collision).
  5. MinimaxAgent.select_action root comparison always maximizes (was inverting for O).
  6. Illegal move update uses done=True to avoid broken bootstrap.
  7. Optimistic init set to 0.0 — positive init was negated during bootstrap, biasing all early updates negative.
"""

from __future__ import annotations

import time
import math
import random
import copy
import joblib
from pathlib import Path
from collections import defaultdict
from typing import Optional, List, Dict, Tuple, Any

import streamlit as st


# ===========================================================================
# PICKLE-SAFE DEFAULT FACTORY
# ===========================================================================

def _zero_default() -> float:
    """Neutral initialisation.

    Previously 0.15 (optimistic). After the zero-sum fix, the bootstrap
    for an unseen next-state becomes reward + gamma * (-0.15), which
    introduces a systematic negative bias into every non-terminal update
    from the very first episode. 0.0 is neutral and avoids that bias.
    Exploration is already handled by epsilon-greedy; optimistic init is
    not needed and actively harmful here.
    """
    return 0.0


# ===========================================================================
# ENVIRONMENT
# ===========================================================================

class TicTacToeEnv:
    REWARD_WIN: float = 2.0
    REWARD_DRAW: float = 0.5
    REWARD_LOSS: float = -2.0
    REWARD_ILLEGAL: float = -1.0   # Changed: -10 caused exploding negative targets
    REWARD_STEP: float = 0.0

    WIN_LINES: Tuple[Tuple[int, int, int], ...] = (
        (0, 1, 2), (3, 4, 5), (6, 7, 8),
        (0, 3, 6), (1, 4, 7), (2, 5, 8),
        (0, 4, 8), (2, 4, 6),
    )

    _SYMMETRIES: Tuple[Tuple[int, ...], ...] = (
        (0, 1, 2, 3, 4, 5, 6, 7, 8),
        (2, 5, 8, 1, 4, 7, 0, 3, 6),
        (8, 7, 6, 5, 4, 3, 2, 1, 0),
        (6, 3, 0, 7, 4, 1, 8, 5, 2),
        (2, 1, 0, 5, 4, 3, 8, 7, 6),
        (6, 7, 8, 3, 4, 5, 0, 1, 2),
        (0, 3, 6, 1, 4, 7, 2, 5, 8),
        (8, 5, 2, 7, 4, 1, 6, 3, 0),
    )

    def __init__(self) -> None:
        self.board: List[int] = [0] * 9
        self.current_player: int = 1
        self.done: bool = False
        self._winner: Optional[int] = None

    def reset(self, first_player: int = 1) -> Tuple[tuple, Dict]:
        self.board = [0] * 9
        self.current_player = first_player
        self.done = False
        self._winner = None
        return self._get_state(), {}

    def step(self, action: int) -> Tuple[tuple, float, bool, bool, Dict]:
        if self.done:
            raise RuntimeError("Episode finished. Call reset().")

        if self.board[action] != 0:
            return self._get_state(), self.REWARD_ILLEGAL, False, False, {"illegal": True}

        self.board[action] = self.current_player
        winner = self.check_winner()
        terminated = False
        reward = self.REWARD_STEP

        if winner == self.current_player:
            reward = self.REWARD_WIN
            terminated = True
            self._winner = winner
        elif winner == -self.current_player:
            reward = self.REWARD_LOSS
            terminated = True
            self._winner = winner
        elif not self.available_actions():
            reward = self.REWARD_DRAW
            terminated = True
            self._winner = 0

        self.done = terminated
        if not terminated:
            self.current_player = -self.current_player

        info = {"winner": self._winner, "board": copy.copy(self.board), "illegal": False}
        return self._get_state(), reward, terminated, False, info

    def render(self) -> str:
        symbols = {0: ".", 1: "X", -1: "O"}
        rows = []
        for r in range(3):
            row_str = " | ".join(symbols[self.board[3 * r + c]] for c in range(3))
            rows.append(f" {row_str} ")
        return "\n-----------\n".join(rows)

    def available_actions(self) -> List[int]:
        return [i for i, v in enumerate(self.board) if v == 0]

    def check_winner(self) -> Optional[int]:
        for a, b, c in self.WIN_LINES:
            s = self.board[a] + self.board[b] + self.board[c]
            if s == 3:
                return 1
            if s == -3:
                return -1
        if not self.available_actions():
            return 0
        return None

    def winning_line(self) -> Optional[Tuple[int, int, int]]:
        for a, b, c in self.WIN_LINES:
            s = self.board[a] + self.board[b] + self.board[c]
            if s == 3 or s == -3:
                return (a, b, c)
        return None

    def _get_state(self) -> tuple:
        return tuple(self.board)

    @staticmethod
    def canonical_state(state: tuple) -> tuple:
        best = state
        for sym in TicTacToeEnv._SYMMETRIES:
            transformed = tuple(state[sym[i]] for i in range(9))
            if transformed < best:
                best = transformed
        return best

    @staticmethod
    def invert_state(state: tuple) -> tuple:
        return tuple(-v for v in state)

    def clone(self) -> "TicTacToeEnv":
        env = TicTacToeEnv()
        env.board = copy.copy(self.board)
        env.current_player = self.current_player
        env.done = self.done
        env._winner = self._winner
        return env


# ===========================================================================
# TACTICAL HELPERS
# ===========================================================================

WIN_LINES = TicTacToeEnv.WIN_LINES
CORNERS = (0, 2, 6, 8)
CENTER = 4


def count_two_in_a_row_threats(board: List[int], player: int) -> int:
    threats = 0
    for a, b, c in WIN_LINES:
        vals = [board[a], board[b], board[c]]
        if vals.count(player) == 2 and vals.count(0) == 1:
            threats += 1
    return threats


def is_fork(board: List[int], player: int) -> bool:
    return count_two_in_a_row_threats(board, player) >= 2


def find_winning_move(board: List[int], player: int, available: List[int]) -> Optional[int]:
    for a, b, c in WIN_LINES:
        cells = [a, b, c]
        vals = [board[i] for i in cells]
        if vals.count(player) == 2 and vals.count(0) == 1:
            empty = cells[vals.index(0)]
            if empty in available:
                return empty
    return None


def compute_shaping_reward(
    board_before: List[int],
    board_after: List[int],
    action: int,
    mover: int,
) -> float:
    opponent = -mover
    shaping = 0.0

    was_empty_board = sum(1 for v in board_before if v != 0) == 0
    if was_empty_board and action == CENTER:
        shaping += 0.05
    elif was_empty_board and action in CORNERS:
        shaping += 0.02

    blocked_action = find_winning_move(
        board_before, opponent,
        [i for i, v in enumerate(board_before) if v == 0]
    )
    if blocked_action == action:
        shaping += 0.05

    if is_fork(board_after, mover) and not is_fork(board_before, mover):
        shaping += 0.06

    opp_could_fork_before = False
    test_board = list(board_before)
    for empty_idx in [i for i, v in enumerate(board_before) if v == 0]:
        test_board[empty_idx] = opponent
        if is_fork(test_board, opponent):
            opp_could_fork_before = True
        test_board[empty_idx] = 0
        if opp_could_fork_before:
            break

    opp_can_fork_after = False
    test_board = list(board_after)
    for empty_idx in [i for i, v in enumerate(board_after) if v == 0]:
        test_board[empty_idx] = opponent
        if is_fork(test_board, opponent):
            opp_can_fork_after = True
        test_board[empty_idx] = 0
        if opp_can_fork_after:
            break

    if opp_could_fork_before and not opp_can_fork_after:
        shaping += 0.04

    threats_before = count_two_in_a_row_threats(board_before, mover)
    threats_after = count_two_in_a_row_threats(board_after, mover)
    if threats_after > threats_before:
        shaping += 0.02

    return shaping


# ===========================================================================
# AGENTS
# ===========================================================================

class QLearningAgent:
    def __init__(
        self,
        alpha: float = 0.3,
        alpha_end: float = 0.05,
        alpha_decay: float = 0.999995,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.99975,
        epsilon_warmup_episodes: int = 0,
        double_q: bool = True,
        optimistic_init: float = 0.0,   # FIX Bug 9: was 0.15; now 0.0 (neutral)
    ) -> None:
        self.alpha = alpha
        self.alpha_start = alpha
        self.alpha_end = alpha_end
        self.alpha_decay = alpha_decay
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.epsilon_warmup_episodes = epsilon_warmup_episodes
        self.double_q = double_q
        self.optimistic_init = optimistic_init
        self.q_a: Dict[tuple, Dict[int, float]] = defaultdict(
            lambda: defaultdict(_zero_default)
        )
        self.q_b: Optional[Dict[tuple, Dict[int, float]]] = (
            defaultdict(lambda: defaultdict(_zero_default)) if double_q else None
        )
        self.episode_count: int = 0
        self.total_updates: int = 0

    def _process_state(self, state: tuple, player: int = 1) -> tuple:
        """Perspective inversion only — no symmetry canonicalization.
        When O acts, flip board signs so the table always indexes
        'my marks = +1, opponent marks = -1'."""
        if player == -1:
            state = TicTacToeEnv.invert_state(state)

        state = TicTacToeEnv.canonical_state(state)

        return state

    def select_action(
        self,
        state: tuple,
        available_actions: List[int],
        player: int = 1,
        training: bool = True,
    ) -> int:
        if not available_actions:
            raise ValueError("No available actions.")
        # Opening Book

        if sum(abs(x) for x in state) == 0:

            if 4 in available_actions:
                return 4

            corners = [0, 2, 6, 8]

            valid_corners = [
                c for c in corners
                if c in available_actions
            ]

            if valid_corners:
                return random.choice(valid_corners)



        # ===== Tactical Layer =====

        board = list(state)

        # Win immediately if possible
        win_move = find_winning_move(
            board,
            player,
            available_actions
        )

        if win_move is not None:
            return win_move

        # Block opponent win
        block_move = find_winning_move(
            board,
            -player,
            available_actions
        )

        if block_move is not None:
            return block_move


        if training and random.random() < self.epsilon:
            return random.choice(available_actions)
        return self._greedy_action(state, available_actions, player)

    def _greedy_action(
        self, state: tuple, available_actions: List[int], player: int = 1
    ) -> int:
        key = self._process_state(state, player)
        if self.double_q:
            q_vals = {
                a: (self.q_a[key][a] + self.q_b[key][a]) / 2.0
                for a in available_actions
            }
        else:
            q_vals = {a: self.q_a[key][a] for a in available_actions}
        max_q = max(q_vals.values())
        best = [a for a, q in q_vals.items() if math.isclose(q, max_q, rel_tol=1e-9)]
        return random.choice(best)

    def get_q_values(
        self, state: tuple, available_actions: List[int], player: int = 1
    ) -> Dict[int, float]:
        key = self._process_state(state, player)
        if self.double_q:
            return {
                a: (self.q_a[key][a] + self.q_b[key][a]) / 2.0
                for a in available_actions
            }
        return {a: self.q_a[key][a] for a in available_actions}

    def get_top_k_moves(
        self,
        state: tuple,
        available_actions: List[int],
        player: int = 1,
        k: int = 3,
    ):
        q_vals = self.get_q_values(state, available_actions, player)
        ranked = sorted(q_vals.items(), key=lambda kv: kv[1], reverse=True)
        return ranked[:k]

    def get_confidence(
        self, state: tuple, available_actions: List[int], player: int = 1
    ) -> float:
        if len(available_actions) <= 1:
            return 1.0
        q_vals = list(self.get_q_values(state, available_actions, player).values())
        max_q, min_q = max(q_vals), min(q_vals)
        denom = abs(max_q) + abs(min_q) + 1e-8
        return float(min(max((max_q - min_q) / denom, 0.0), 1.0))

    def update(
        self,
        state: tuple,
        action: int,
        reward: float,
        next_state: tuple,
        next_available: List[int],
        done: bool,
        player: int = 1,
        next_player: Optional[int] = None,
    ) -> None:
        """
        Zero-sum Bellman backup.

        target = reward + gamma * (-max_a Q(next_state, a))

        The negation is mandatory: Q values are stored from the acting
        player's perspective. After the current player moves, it is the
        opponent's turn. The opponent's best outcome is the current
        player's worst outcome, so we negate.

        Terminal transitions use target = reward with no bootstrap.
        """
        if next_player is None:
            next_player = -player

        key = self._process_state(state, player)
        next_key = self._process_state(next_state, next_player)

        if self.double_q:
            self._double_q_update(key, action, reward, next_key, next_available, done)
        else:
            self._single_q_update(key, action, reward, next_key, next_available, done)
        self.total_updates += 1

    def _single_q_update(
        self,
        key: tuple,
        action: int,
        reward: float,
        next_key: tuple,
        next_available: List[int],
        done: bool,
    ) -> None:
        if done or not next_available:
            target = reward
        else:
            best_next = max(self.q_a[next_key][a] for a in next_available)
            target = reward + self.gamma * (-best_next)          # zero-sum negation
        self.q_a[key][action] += self.alpha * (target - self.q_a[key][action])

    def _double_q_update(
        self,
        key: tuple,
        action: int,
        reward: float,
        next_key: tuple,
        next_available: List[int],
        done: bool,
    ) -> None:
        if done or not next_available:
            target = reward
            if random.random() < 0.5:
                self.q_a[key][action] += self.alpha * (target - self.q_a[key][action])
            else:
                self.q_b[key][action] += self.alpha * (target - self.q_b[key][action])
            return

        if random.random() < 0.5:
            best = max(next_available, key=lambda a: self.q_a[next_key][a])
            target = reward + self.gamma * (-self.q_b[next_key][best])  # zero-sum
            self.q_a[key][action] += self.alpha * (target - self.q_a[key][action])
        else:
            best = max(next_available, key=lambda a: self.q_b[next_key][a])
            target = reward + self.gamma * (-self.q_a[next_key][best])  # zero-sum
            self.q_b[key][action] += self.alpha * (target - self.q_b[key][action])

    def decay_epsilon(self, episode_idx: int, total_episodes: int):
        if episode_idx <= self.epsilon_warmup_episodes:
            self.epsilon = self.epsilon_start
        else:
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def decay_alpha(self):
        self.alpha = max(self.alpha_end, self.alpha * self.alpha_decay)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "q_a": dict(self.q_a),
            "q_b": dict(self.q_b) if self.q_b is not None else None,
            "hyperparams": {
                "alpha": self.alpha,
                "alpha_start": self.alpha_start,
                "alpha_end": self.alpha_end,
                "alpha_decay": self.alpha_decay,
                "gamma": self.gamma,
                "epsilon": self.epsilon,
                "epsilon_start": self.epsilon_start,
                "epsilon_end": self.epsilon_end,
                "epsilon_decay": self.epsilon_decay,
                "epsilon_warmup_episodes": self.epsilon_warmup_episodes,
                "double_q": self.double_q,
                "optimistic_init": self.optimistic_init,
            },
            "stats": {
                "episode_count": self.episode_count,
                "total_updates": self.total_updates,
                "q_table_size": len(self.q_a),
            },
        }
        joblib.dump(payload, path, compress=3)

    @classmethod
    def load(cls, path: str) -> "QLearningAgent":
        payload = joblib.load(path)
        hp = payload["hyperparams"]
        agent = cls(
            alpha=hp.get("alpha_start", hp.get("alpha", 0.3)),
            alpha_end=hp.get("alpha_end", 0.05),
            alpha_decay=hp.get("alpha_decay", 0.999995),
            gamma=hp["gamma"],
            epsilon_start=hp["epsilon_start"],
            epsilon_end=hp["epsilon_end"],
            epsilon_decay=hp["epsilon_decay"],
            epsilon_warmup_episodes=hp.get("epsilon_warmup_episodes", 0),
            double_q=hp["double_q"],
            optimistic_init=hp.get("optimistic_init", 0.0),
        )
        agent.alpha = hp.get("alpha", agent.alpha)
        agent.epsilon = hp["epsilon"]
        agent.q_a = defaultdict(lambda: defaultdict(_zero_default))
        for k, v in payload["q_a"].items():
            agent.q_a[k] = defaultdict(_zero_default, v)
        if payload["q_b"] is not None:
            agent.q_b = defaultdict(lambda: defaultdict(_zero_default))
            for k, v in payload["q_b"].items():
                agent.q_b[k] = defaultdict(_zero_default, v)
        agent.episode_count = payload["stats"]["episode_count"]
        agent.total_updates = payload["stats"]["total_updates"]
        return agent

    def stats(self) -> Dict[str, Any]:
        return {
            "epsilon": round(self.epsilon, 6),
            "alpha": round(self.alpha, 6),
            "gamma": self.gamma,
            "q_table_size": len(self.q_a),
            "episode_count": self.episode_count,
            "total_updates": self.total_updates,
            "double_q": self.double_q,
            "use_symmetry": False,
        }


class MinimaxAgent:
    """
    FIX Bug 4 + Bug 5:

    Bug 4 — Cache key missing max_player.
    BEFORE: cache_key = (tuple(board), int(is_maximising))
    AFTER:  cache_key = (tuple(board), int(is_maximising), max_player)

    When MinimaxAgent is reused across episodes where it plays as X in
    some and O in others, the same board position with the same
    is_maximising flag but different max_player values would return a
    cached value computed for the wrong player. This silently made the
    agent play suboptimally or even self-destructively as O.

    Bug 5 — select_action root comparison wrong for O.
    BEFORE: best_score = -inf if acting_player==1 else +inf
            then: score > best_score for X, score < best_score for O
    This made O pick the *minimum* minimax score at the root, which is
    the worst move for O (since _minimax returns scores from max_player's
    perspective and max_player == acting_player).
    AFTER: always maximize at root regardless of acting_player, because
    _minimax already returns value from acting_player's POV.
    """

    def __init__(self, player: int = 1) -> None:
        self.player = player
        self._cache: Dict[Tuple[tuple, int, int], float] = {}

    def select_action(
        self,
        state: tuple,
        available_actions: List[int],
        player: Optional[int] = None,
        **kwargs,
    ) -> int:
        acting_player = player if player is not None else self.player
        board = list(state)
        best_action = available_actions[0]
        best_score = -math.inf          # FIX Bug 5: always maximize at root

        for action in available_actions:
            board[action] = acting_player
            score = self._minimax(
                board, 0, -math.inf, math.inf,
                False,                  # after root move, opponent minimizes
                acting_player,
            )
            board[action] = 0
            if score > best_score:      # FIX Bug 5: always take the max
                best_score = score
                best_action = action

        return best_action

    def _minimax(
        self,
        board: List[int],
        depth: int,
        alpha: float,
        beta: float,
        is_maximising: bool,
        max_player: int,
    ) -> float:
        # FIX Bug 4: include max_player in cache key
        cache_key = (tuple(board), int(is_maximising), max_player)
        if cache_key in self._cache:
            return self._cache[cache_key]

        winner = self._check_winner(board)
        if winner == max_player:
            result = 10 - depth
            self._cache[cache_key] = result
            return result
        if winner == -max_player:
            result = depth - 10
            self._cache[cache_key] = result
            return result
        if winner == 0:
            self._cache[cache_key] = 0
            return 0

        empty = [i for i, v in enumerate(board) if v == 0]
        if is_maximising:
            best = -math.inf
            for action in empty:
                board[action] = max_player
                val = self._minimax(board, depth + 1, alpha, beta, False, max_player)
                board[action] = 0
                best = max(best, val)
                alpha = max(alpha, best)
                if beta <= alpha:
                    break
        else:
            best = math.inf
            for action in empty:
                board[action] = -max_player
                val = self._minimax(board, depth + 1, alpha, beta, True, max_player)
                board[action] = 0
                best = min(best, val)
                beta = min(beta, best)
                if beta <= alpha:
                    break

        self._cache[cache_key] = best
        return best

    @staticmethod
    def _check_winner(board: List[int]) -> Optional[int]:
        lines = (
            (0, 1, 2), (3, 4, 5), (6, 7, 8),
            (0, 3, 6), (1, 4, 7), (2, 5, 8),
            (0, 4, 8), (2, 4, 6),
        )
        for a, b, c in lines:
            s = board[a] + board[b] + board[c]
            if s == 3:
                return 1
            if s == -3:
                return -1
        if all(v != 0 for v in board):
            return 0
        return None


class RandomAgent:
    def __init__(self, player: int = -1) -> None:
        self.player = player

    def select_action(self, state, available_actions, player=None, **kwargs) -> int:
        return random.choice(available_actions)


class RuleBasedAgent:
    CORNERS = [0, 2, 6, 8]
    EDGES = [1, 3, 5, 7]
    CENTER = 4

    def __init__(self, player: int = -1) -> None:
        self.player = player

    def select_action(self, state, available_actions, player=None, **kwargs) -> int:
        acting = player if player is not None else self.player
        opponent = -acting
        win = self._find_winning_move(list(state), acting, available_actions)
        if win is not None:
            return win
        block = self._find_winning_move(list(state), opponent, available_actions)
        if block is not None:
            return block
        if self.CENTER in available_actions:
            return self.CENTER
        corners = [c for c in self.CORNERS if c in available_actions]
        if corners:
            return random.choice(corners)
        edges = [e for e in self.EDGES if e in available_actions]
        if edges:
            return random.choice(edges)
        return random.choice(available_actions)

    @staticmethod
    def _find_winning_move(board, player, available):
        lines = (
            (0, 1, 2), (3, 4, 5), (6, 7, 8),
            (0, 3, 6), (1, 4, 7), (2, 5, 8),
            (0, 4, 8), (2, 4, 6),
        )
        for a, b, c in lines:
            cells = [a, b, c]
            vals = [board[i] for i in cells]
            if vals.count(player) == 2 and vals.count(0) == 1:
                empty = cells[vals.index(0)]
                if empty in available:
                    return empty
        return None


# ===========================================================================
# TRAINING
# ===========================================================================

def run_training(
    episodes: int = 100_000,
    progress_bar=None,
    status_text=None,
) -> QLearningAgent:
    """
    Fixes applied in the training loop:

    Bug 2 fix — last_transition retrospective update only fires when the
    OPPONENT terminated the episode, not the learner.

    Bug 6 fix — illegal-move penalty uses done=True so there is no
    broken bootstrap from a self-referential next_key. The penalty is
    also now just REWARD_ILLEGAL (–1.0, same magnitude as a loss) rather
    than –10, which was causing Q-values to diverge far outside [–1, 1].

    Bug 8 fix — when an illegal move is penalized we pass done=True and
    do not pass next_player=player (which was producing a self-loop
    bootstrap). The update is a pure terminal penalty: target = -1.0.
    """
    warmup_frac = 0.02
    warmup_episodes = max(1, int(episodes * warmup_frac))

    agent = QLearningAgent(
        alpha=0.3,
        alpha_end=0.05,
        alpha_decay=0.999995,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.99975,
        epsilon_warmup_episodes=warmup_episodes,
        double_q=True,
        optimistic_init=0.0,        # FIX Bug 9: neutral init
    )
    env = TicTacToeEnv()
    # FIX Bug 4/10: create fresh MinimaxAgent instances per training run.
    # The shared cache now correctly keys on max_player, so reuse is safe,
    # but we keep one instance per role for clarity.
    rule_opp = RuleBasedAgent()
    minimax_opp = MinimaxAgent()

    OPPONENT_RULE = "rule"
    OPPONENT_MINIMAX = "minimax"

    for ep in range(1, episodes + 1):
        r = random.random()

        if r < 0.30:
            opponent_type = "selfplay"
        elif r < 0.60:
            opponent_type = OPPONENT_RULE
        else:
            opponent_type = OPPONENT_MINIMAX

        learner_side = 1 if ep % 2 == 0 else -1
        state, _ = env.reset(first_player=1)
        last_transition = None      # (state, action, player) of learner's last non-terminal move
        done = False
        info: Dict = {}

        while not done:
            player = env.current_player
            available = env.available_actions()
            board_before = list(env.board)

            if player == learner_side:
                # --- Learner's turn ---
                # FIX Bug 6: pick only from genuinely available cells.
                # If greedy action is illegal (epsilon=0 edge case after table
                # corruption), fall back to a random legal move instead of
                # looping and potentially re-executing an illegal action.
                action = agent.select_action(state, available, player=player, training=True)

                # Ensure the chosen action is legal (it must be, since
                # select_action receives `available`). If somehow it is not,
                # penalize and pick a random legal move.
                next_state, reward, terminated, _, info = env.step(action)

                if info.get("illegal"):
                    # FIX Bug 6 + Bug 8: terminal penalty, no bootstrap.
                    agent.update(
                        state, action, TicTacToeEnv.REWARD_ILLEGAL,
                        state, [],          # empty next_available forces done-branch
                        True,               # done=True: no bootstrap
                        player=player,
                        next_player=-player,
                    )
                    # Fall back to a random legal move to keep episode going
                    action = random.choice(available)
                    next_state, reward, terminated, _, info = env.step(action)

                is_learner_move = True

                if not terminated:
                    shaping = compute_shaping_reward(
                        board_before, list(env.board), action, player
                    )
                    reward = reward + shaping
            else:
                # --- Opponent's turn ---
                if opponent_type == "selfplay":

                    action = agent.select_action(
                        state,
                        available,
                        player=player,
                        training=False
                    )

                elif opponent_type == OPPONENT_RULE:

                    action = rule_opp.select_action(
                        state,
                        available,
                        player=player
                    )

                else:

                    action = minimax_opp.select_action(
                        state,
                        available,
                        player=player
                    )

                next_state, reward, terminated, _, info = env.step(action)

                is_learner_move = False

            # FIX Bug 2: retrospective last_transition update ONLY when the
            # opponent terminated. When the learner terminates, the direct
            # update in the is_learner_move block below handles it.
            if terminated and not is_learner_move and last_transition is not None:
                prev_state, prev_action, prev_player = last_transition
                w = info["winner"]
                if w == -prev_player:
                    prev_reward = TicTacToeEnv.REWARD_LOSS
                elif w == 0:
                    prev_reward = TicTacToeEnv.REWARD_DRAW
                else:
                    prev_reward = TicTacToeEnv.REWARD_WIN
                agent.update(
                    prev_state, prev_action, prev_reward,
                    next_state, [],
                    True,
                    player=prev_player,
                    next_player=-prev_player,
                )

            if is_learner_move:
                if not terminated:
                    agent.update(
                        state, action, reward,
                        next_state, env.available_actions(),
                        False,
                        player=player,
                        next_player=-player,
                    )
                    last_transition = (state, action, player)
                else:
                    agent.update(
                        state, action, reward,
                        next_state, [],
                        True,
                        player=player,
                        next_player=-player,
                    )
                    last_transition = None  # learner ended it — no retrospective needed

            state = next_state
            done = terminated

        agent.decay_epsilon(ep, episodes)
        agent.decay_alpha()
        agent.episode_count += 1

        if progress_bar and ep % max(1, episodes // 200) == 0:
            progress_bar.progress(ep / episodes)
        if status_text and ep % max(1, episodes // 50) == 0:
            status_text.text(
                f"Episode {ep:,} / {episodes:,} | ε={agent.epsilon:.4f} | "
                f"α={agent.alpha:.4f} | States={len(agent.q_a):,} | opp={opponent_type}"
            )

    return agent


# ===========================================================================
# PERFORMANCE EVALUATION
# ===========================================================================

def evaluate_agent(
    agent: QLearningAgent, n_games: int = 1000
) -> Dict[str, Dict[str, Any]]:
    opponents = {
        "RandomAgent": RandomAgent(),
        "RuleBasedAgent": RuleBasedAgent(),
        "MinimaxAgent": MinimaxAgent(),
    }
    results: Dict[str, Dict[str, Any]] = {}
    env = TicTacToeEnv()

    for opp_name, opponent in opponents.items():
        wins = draws = losses = 0
        for g in range(n_games):
            agent_side = 1 if g % 2 == 0 else -1
            # FIX Bug 3: always first_player=1 (X always moves first in
            # real Tic-Tac-Toe). The old code tied first_player to agent_side,
            # which made O-agent games start from an impossible board state.
            state, _ = env.reset(first_player=1)
            done = False
            info: Dict = {}
            while not done:
                player = env.current_player
                available = env.available_actions()
                if player == agent_side:
                    action = agent.select_action(
                        state, available, player=player, training=False
                    )
                else:
                    action = opponent.select_action(
                        state, available, player=player
                    )
                state, _, done, _, info = env.step(action)
            w = info.get("winner")
            if w == agent_side:
                wins += 1
            elif w == 0:
                draws += 1
            else:
                losses += 1

        results[opp_name] = {
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "win_rate": wins / n_games,
            "non_loss_rate": (wins + draws) / n_games,
        }
    return results


# ===========================================================================
# STREAMLIT APP
# ===========================================================================

st.set_page_config(
    page_title="Tic-Tac-Toe RL",
    page_icon="🎮",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
div[data-testid="stButton"] > button {
    width: 100% !important;
    height: 112px !important;
    font-size: 3.1rem !important;
    font-weight: 800 !important;
    border-radius: 18px !important;
    border: 2px solid #D9D7CC !important;
    background: #FFFFFF !important;
    color: #2C2C2A !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease, background 0.15s ease !important;
    line-height: 1 !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04) !important;
    position: relative !important;
}
div[data-testid="stButton"] > button:hover:not(:disabled) {
    border-color: #7F77DD !important;
    background: #F2F1FE !important;
    transform: translateY(-2px) scale(1.035) !important;
    box-shadow: 0 6px 14px rgba(127,119,221,0.22) !important;
}
div[data-testid="stButton"] > button:active:not(:disabled) {
    transform: scale(0.97) !important;
}
div[data-testid="stButton"] > button:disabled {
    opacity: 1 !important;
    cursor: default !important;
}
.x-cell button {
    color: #E0512F !important;
    border-color: #F3BFA9 !important;
    background: linear-gradient(135deg, #FFF4F0 0%, #FCE7E0 100%) !important;
}
.x-cell button:hover:not(:disabled) { border-color: #E0512F !important; }
.o-cell button {
    color: #1A5DA6 !important;
    border-color: #AFD2F2 !important;
    background: linear-gradient(135deg, #F0F7FE 0%, #E2EFFB 100%) !important;
}
.o-cell button:hover:not(:disabled) { border-color: #1A5DA6 !important; }
@keyframes popIn {
    0%   { transform: scale(0.4); opacity: 0; }
    60%  { transform: scale(1.12); opacity: 1; }
    100% { transform: scale(1); opacity: 1; }
}
.cell-recent button {
    animation: popIn 0.32s cubic-bezier(0.34, 1.56, 0.64, 1) !important;
    box-shadow: 0 0 0 3px rgba(127,119,221,0.55), 0 4px 10px rgba(0,0,0,0.08) !important;
}
@keyframes winGlow {
    0%, 100% { box-shadow: 0 0 0 3px rgba(34,197,94,0.45), 0 0 18px rgba(34,197,94,0.35); }
    50%      { box-shadow: 0 0 0 5px rgba(34,197,94,0.75), 0 0 28px rgba(34,197,94,0.55); }
}
.cell-winning button {
    border-color: #22C55E !important;
    animation: winGlow 1.1s ease-in-out infinite !important;
}
.winner-banner {
    text-align: center;
    font-size: 1.6rem;
    font-weight: 700;
    padding: 16px 20px;
    border-radius: 14px;
    margin: 12px 0 18px 0;
    animation: popIn 0.35s ease;
}
.stat-box {
    background: #F7F5F0;
    border: 1px solid #D3D1C7;
    border-radius: 12px;
    padding: 14px 10px;
    text-align: center;
}
.stat-num { font-size: 2rem; font-weight: 700; }
.stat-lbl { font-size: 0.78rem; color: #5F5E5A; margin-top: 3px; }
.section-title {
    font-size: 0.78rem;
    font-weight: 600;
    color: #888780;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin: 14px 0 6px 0;
}
.ai-panel {
    background: linear-gradient(135deg, #F8F7FF 0%, #F1F0FE 100%);
    border: 1px solid #DAD7F7;
    border-radius: 16px;
    padding: 16px 18px;
    margin-top: 14px;
}
.ai-panel-title {
    font-size: 0.8rem;
    font-weight: 700;
    color: #4C44A8;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 8px;
}
.ai-explanation {
    font-size: 0.92rem;
    color: #38356B;
    background: #FFFFFF;
    border-radius: 10px;
    padding: 10px 12px;
    margin-top: 8px;
    border-left: 3px solid #7F77DD;
}
.candidate-row {
    display: flex;
    justify-content: space-between;
    padding: 5px 0;
    font-size: 0.88rem;
    border-bottom: 1px solid #E9E7FB;
}
.candidate-row:last-child { border-bottom: none; }
.candidate-rank { font-weight: 700; color: #7F77DD; width: 22px; }
</style>
""", unsafe_allow_html=True)

MODEL_PATH = "models/q_table.pkl"
CELL_NAMES = {
    0: "top-left", 1: "top-center", 2: "top-right",
    3: "middle-left", 4: "center", 5: "middle-right",
    6: "bottom-left", 7: "bottom-center", 8: "bottom-right",
}

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def init_state():
    defaults = {
        "board": [0] * 9,
        "current_player": 1,
        "done": False,
        "winner": None,
        "mode": "Human vs AI",
        "human_player": 1,
        "stats": {"wins": 0, "draws": 0, "losses": 0},
        "show_qvalues": False,
        "game_log": [],
        "ai_last_action": None,
        "ai_last_confidence": None,
        "ai_top_moves": [],
        "eval_results": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset_game():
    st.session_state.board = [0] * 9
    st.session_state.current_player = 1
    st.session_state.done = False
    st.session_state.winner = None
    st.session_state.game_log = []
    st.session_state.ai_last_action = None
    st.session_state.ai_last_confidence = None
    st.session_state.ai_top_moves = []


# ---------------------------------------------------------------------------
# Agent loaders
# ---------------------------------------------------------------------------

@st.cache_resource
def load_rl_agent():
    if not Path(MODEL_PATH).exists():
        return None
    agent = QLearningAgent.load(MODEL_PATH)
    agent.epsilon = 0.0
    return agent


@st.cache_resource
def load_minimax():
    return MinimaxAgent()


# ---------------------------------------------------------------------------
# Game helpers
# ---------------------------------------------------------------------------

def get_env_from_state() -> TicTacToeEnv:
    env = TicTacToeEnv()
    env.board = list(st.session_state.board)
    env.current_player = st.session_state.current_player
    env.done = st.session_state.done
    return env


def apply_move(action: int):
    env = get_env_from_state()
    if env.done or env.board[action] != 0:
        return
    mover = env.current_player
    _, reward, terminated, _, info = env.step(action)
    st.session_state.board = list(env.board)
    st.session_state.current_player = env.current_player
    st.session_state.done = terminated
    st.session_state.winner = info.get("winner")
    sym = "✕" if mover == 1 else "○"
    st.session_state.game_log.append(f"{sym} → cell {action}")
    if terminated:
        w = st.session_state.winner
        hp = st.session_state.human_player
        if w == hp:
            st.session_state.stats["wins"] += 1
        elif w == 0:
            st.session_state.stats["draws"] += 1
        else:
            st.session_state.stats["losses"] += 1


def ai_move(agent):
    env = get_env_from_state()
    if env.done:
        return
    available = env.available_actions()
    if not available:
        return
    player = env.current_player
    state_tuple = tuple(env.board)
    action = agent.select_action(state_tuple, available, player=player, training=False)

    conf = None
    top_moves = []
    if hasattr(agent, "get_confidence"):
        conf = agent.get_confidence(state_tuple, available, player=player)
    if hasattr(agent, "get_top_k_moves"):
        top_moves = agent.get_top_k_moves(state_tuple, available, player=player, k=3)

    st.session_state.ai_last_action = action
    st.session_state.ai_last_confidence = conf
    st.session_state.ai_top_moves = top_moves
    apply_move(action)


def build_ai_explanation(top_moves, chosen_action: int) -> str:
    if not top_moves:
        return "AI selected a move."
    chosen_label = CELL_NAMES.get(chosen_action, f"cell {chosen_action}")
    best_q = top_moves[0][1]
    if len(top_moves) > 1:
        margin = top_moves[0][1] - top_moves[1][1]
        margin_txt = (
            f" It led the next-best option by {margin:+.2f} in estimated value, "
            f"so the choice was fairly clear."
            if margin > 0.05
            else f" The margin over the next-best option was very small ({margin:+.2f}), "
            f"meaning several moves looked roughly equally good."
        )
    else:
        margin_txt = " It was the only legal move available."
    return (
        f"AI preferred the {chosen_label} cell because it had the highest estimated "
        f"future reward (Q ≈ {best_q:+.2f}).{margin_txt}"
    )


# ---------------------------------------------------------------------------
# Board rendering
# ---------------------------------------------------------------------------

SYMBOLS = {0: "", 1: "✕", -1: "○"}


def render_board(agent=None):
    board = st.session_state.board
    mode = st.session_state.mode
    human_player = st.session_state.human_player

    q_vals = {}
    if agent and st.session_state.show_qvalues and not st.session_state.done:
        env = get_env_from_state()
        available = env.available_actions()
        if available:
            ai_p = -human_player if mode == "Human vs AI" else env.current_player
            q_vals = agent.get_q_values(tuple(board), available, player=ai_p)

    last_ai = st.session_state.get("ai_last_action")
    env_check = get_env_from_state()
    winning_cells = set(env_check.winning_line() or ())

    for row in range(3):
        cols = st.columns(3, gap="small")
        for col in range(3):
            idx = row * 3 + col
            cell_val = board[idx]
            symbol = SYMBOLS[cell_val]

            is_human_turn = st.session_state.current_player == human_player
            disabled = (
                st.session_state.done
                or cell_val != 0
                or mode != "Human vs AI"
                or not is_human_turn
            )

            if idx in q_vals:
                label = f"{symbol or '·'}\n{q_vals[idx]:+.2f}"
            elif symbol:
                label = symbol
            else:
                label = "·" if disabled else " "

            css_classes = []
            if cell_val == 1:
                css_classes.append("x-cell")
            elif cell_val == -1:
                css_classes.append("o-cell")
            if idx == last_ai and cell_val != 0:
                css_classes.append("cell-recent")
            if idx in winning_cells:
                css_classes.append("cell-winning")
            wrapper_class = " ".join(css_classes) if css_classes else "empty-cell"

            with cols[col]:
                st.markdown(f'<div class="{wrapper_class}">', unsafe_allow_html=True)
                clicked = st.button(
                    label,
                    key=f"cell_{idx}",
                    disabled=disabled,
                    help=f"Cell {idx}"
                    + (f" | Q={q_vals[idx]:+.3f}" if idx in q_vals else ""),
                )
                st.markdown("</div>", unsafe_allow_html=True)
                if clicked and not disabled:
                    apply_move(idx)
                    st.rerun()


def render_ai_panel():
    action = st.session_state.get("ai_last_action")
    conf = st.session_state.get("ai_last_confidence")
    top_moves = st.session_state.get("ai_top_moves") or []

    if action is None:
        return

    st.markdown('<div class="ai-panel">', unsafe_allow_html=True)
    st.markdown('<div class="ai-panel-title">🤖 AI Insight</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        cell_label = CELL_NAMES.get(action, f"cell {action}")
        st.markdown(f"**Move chosen:** {cell_label} (cell {action})")
    with col_b:
        if conf is not None:
            st.markdown(f"**Confidence:** {int(conf * 100)}%")
            st.progress(conf)

    if top_moves:
        st.markdown("**Top candidate moves (by Q-value):**")
        for rank, (a, q) in enumerate(top_moves, start=1):
            label = CELL_NAMES.get(a, f"cell {a}")
            marker = " ← chosen" if a == action else ""
            st.markdown(
                f'<div class="candidate-row">'
                f'<span><span class="candidate-rank">#{rank}</span> {label} (cell {a}){marker}</span>'
                f'<span>Q = {q:+.3f}</span></div>',
                unsafe_allow_html=True,
            )

    explanation = build_ai_explanation(top_moves, action)
    st.markdown(
        f'<div class="ai-explanation">{explanation}</div>', unsafe_allow_html=True
    )
    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Watch mode
# ---------------------------------------------------------------------------

def watch_game(agent_x, agent_o, delay: float = 0.55):
    env = TicTacToeEnv()
    state, _ = env.reset()
    board_placeholder = st.empty()
    log_placeholder = st.empty()
    moves = []

    done = False
    while not done:
        player = env.current_player
        available = env.available_actions()
        agent = agent_x if player == 1 else agent_o
        action = agent.select_action(state, available, player=player, training=False)
        state, _, done, _, info = env.step(action)
        sym = "✕" if player == 1 else "○"
        moves.append(f"{sym} → cell {action}")

        with board_placeholder.container():
            _render_static_board(env.board, last_action=action)

        log_placeholder.caption(" · ".join(moves[-6:]))
        time.sleep(delay)

    winner = info.get("winner")
    banners = {
        1: (
            "<div class='winner-banner' style='background:#EAF3DE;color:#27500A'>✕ Agent X wins!</div>",
            "success",
        ),
        -1: (
            "<div class='winner-banner' style='background:#FCEBEB;color:#791F1F'>○ Agent O wins!</div>",
            "error",
        ),
        0: (
            "<div class='winner-banner' style='background:#FAEEDA;color:#633806'>🤝 Draw!</div>",
            "warning",
        ),
    }
    html, _ = banners.get(winner, ("<div class='winner-banner'>Game over</div>", "info"))
    st.markdown(html, unsafe_allow_html=True)


def _render_static_board(board: list, last_action: Optional[int] = None):
    sym = {0: "·", 1: "✕", -1: "○"}
    rows = []
    for r in range(3):
        parts = []
        for c in range(3):
            idx = 3 * r + c
            s = sym[board[idx]]
            if idx == last_action:
                s = f"[{s}]"
            parts.append(f"  {s}  ")
        rows.append("|".join(parts))
    divider = "-" * (len(rows[0]))
    board_str = f"\n{divider}\n".join(rows)
    st.code(board_str, language=None)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    init_state()
    rl_agent = load_rl_agent()
    minimax_agent = load_minimax()

    with st.sidebar:
        st.markdown("## 🎮 Tic-Tac-Toe RL")
        st.caption("Double Q-Learning · No Symmetry · RuleBased+Minimax Training")
        st.divider()

        mode = st.selectbox(
            "Game mode",
            ["Human vs AI", "AI vs Random", "AI vs Minimax"],
        )
        st.session_state.mode = mode

        if mode == "Human vs AI":
            choice = st.radio("You play as", ["✕ X  (goes first)", "○ O  (goes second)"])
            st.session_state.human_player = 1 if "X" in choice else -1

        st.session_state.show_qvalues = st.toggle(
            "Show Q-values on board", value=st.session_state.show_qvalues
        )

        st.divider()
        col_r, col_s = st.columns(2)
        with col_r:
            if st.button("🔄 Restart", use_container_width=True):
                reset_game()
                st.rerun()
        with col_s:
            if st.button("🗑️ Stats", use_container_width=True):
                st.session_state.stats = {"wins": 0, "draws": 0, "losses": 0}
                st.rerun()

        st.divider()
        st.markdown(
            "<div class='section-title'>Session statistics</div>",
            unsafe_allow_html=True,
        )
        stats = st.session_state.stats
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(
                f"<div class='stat-box'><div class='stat-num' style='color:#1D9E75'>"
                f"{stats['wins']}</div><div class='stat-lbl'>Wins</div></div>",
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f"<div class='stat-box'><div class='stat-num' style='color:#BA7517'>"
                f"{stats['draws']}</div><div class='stat-lbl'>Draws</div></div>",
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                f"<div class='stat-box'><div class='stat-num' style='color:#A32D2D'>"
                f"{stats['losses']}</div><div class='stat-lbl'>Losses</div></div>",
                unsafe_allow_html=True,
            )

        total = sum(stats.values())
        if total > 0:
            wr = stats["wins"] / total
            st.progress(wr, text=f"Win rate: {wr:.0%}")

        st.divider()
        st.markdown(
            "<div class='section-title'>Model info</div>", unsafe_allow_html=True
        )
        if rl_agent:
            s = rl_agent.stats()
            st.caption(f"States in Q-table: **{s['q_table_size']:,}**")
            st.caption(f"Episodes trained: **{s['episode_count']:,}**")
            st.caption(f"Double Q-Learning: **{s['double_q']}**")
            st.caption(f"Symmetry reduction: **{s['use_symmetry']}**")
            st.caption(f"gamma: **{s['gamma']}** | alpha: **{s['alpha']}**")
        else:
            st.warning("No trained model found.")
            st.caption(f"Expected: `{MODEL_PATH}`")

        st.divider()
        st.markdown(
            "<div class='section-title'>Train model here</div>", unsafe_allow_html=True
        )
        n_ep = st.select_slider(
            "Episodes",
            options=[10_000, 50_000, 100_000, 200_000, 500_000, 1_000_000,2000000,5000000],
            value=200_000,
        )
        st.caption(
            "Training opponents: 40% RuleBased · 60% Minimax (no self-play)."
        )
        if st.button("🚀 Train now", use_container_width=True):
            with st.spinner("Training in progress…"):
                pb = st.progress(0.0)
                st_txt = st.empty()
                trained = run_training(n_ep, progress_bar=pb, status_text=st_txt)
                Path("models").mkdir(exist_ok=True)
                trained.save(MODEL_PATH)
                pb.progress(1.0)
                st_txt.text("✅ Training done. Running automatic evaluation…")
                st.session_state.eval_results = evaluate_agent(trained, n_games=1000)
                st_txt.text("✅ Done! Reload the page to use the new model.")
            st.cache_resource.clear()
            st.success(f"Trained {n_ep:,} episodes. Model saved and evaluated.")
            st.rerun()

        st.divider()
        st.markdown(
            "<div class='section-title'>Evaluate model</div>", unsafe_allow_html=True
        )
        if rl_agent:
            if st.button(
                "📊 Run evaluation (1000 games × 3 opponents)",
                use_container_width=True,
            ):
                with st.spinner("Playing 3,000 evaluation games…"):
                    st.session_state.eval_results = evaluate_agent(
                        rl_agent, n_games=1000
                    )
                st.rerun()
        else:
            st.caption("Train a model first to run evaluation.")

    st.title("Tic-Tac-Toe")

    if st.session_state.get("eval_results"):
        with st.expander(
            "📊 Agent Evaluation Results (1000 games per opponent)", expanded=True
        ):
            for opp_name, res in st.session_state.eval_results.items():
                st.markdown(f"**vs {opp_name}**")
                cols = st.columns(4)
                cols[0].metric("Wins", res["wins"])
                cols[1].metric("Draws", res["draws"])
                cols[2].metric("Losses", res["losses"])
                cols[3].metric("Win rate", f"{res['win_rate']:.1%}")
                st.caption(f"Non-loss rate: {res['non_loss_rate']:.1%}")
                st.progress(res["win_rate"])
                st.divider()

    if mode == "Human vs AI":
        if not rl_agent:
            st.error(
                "No trained model found at `models/q_table.pkl`.\n\n"
                "Use the **Train model here** button in the sidebar to train one now."
            )
            return

        if st.session_state.done:
            w = st.session_state.winner
            hp = st.session_state.human_player
            if w == hp:
                st.markdown(
                    "<div class='winner-banner' style='background:#EAF3DE;color:#27500A'>🎉 You win!</div>",
                    unsafe_allow_html=True,
                )
            elif w == 0:
                st.markdown(
                    "<div class='winner-banner' style='background:#FAEEDA;color:#633806'>🤝 Draw!</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    "<div class='winner-banner' style='background:#FCEBEB;color:#791F1F'>🤖 AI wins!</div>",
                    unsafe_allow_html=True,
                )
        else:
            cp = st.session_state.current_player
            hp = st.session_state.human_player
            sym_map = {1: "✕", -1: "○"}
            if cp == hp:
                st.info(f"Your turn — you are **{sym_map[hp]}**")
            else:
                st.info("AI is thinking… 🤖")

        render_board(rl_agent)

        if st.session_state.show_qvalues and not st.session_state.done:
            st.caption(
                "Numbers on board = AI's Q-value estimate for each empty cell "
                "(higher = AI prefers that cell)."
            )

        render_ai_panel()

        if (
            not st.session_state.done
            and st.session_state.current_player != st.session_state.human_player
        ):
            time.sleep(0.25)
            ai_move(rl_agent)
            st.rerun()

    elif mode == "AI vs Random":
        agent_x = rl_agent if rl_agent else minimax_agent
        st.info("**RL Agent (✕)** vs **Random Agent (○)** — watch the trained agent dominate.")

        delay = st.slider("Move delay (s)", 0.1, 1.5, 0.5, 0.1)
        if st.button("▶ Play a game", use_container_width=True):
            watch_game(agent_x, RandomAgent(), delay=delay)

        st.divider()
        st.markdown(
            "**What to watch for:** The RL agent should win the vast majority of games."
        )

    elif mode == "AI vs Minimax":
        agent_x = rl_agent if rl_agent else minimax_agent
        st.info(
            "**RL Agent (✕)** vs **Minimax Agent (○)** — a draw means near-perfect play."
        )

        delay = st.slider("Move delay (s)", 0.1, 2.0, 0.7, 0.1)
        if st.button("▶ Play a game", use_container_width=True):
            watch_game(agent_x, minimax_agent, delay=delay)

        st.divider()
        st.markdown(
            "**Minimax is a perfect player.** It never loses. A well-trained RL agent "
            "should draw almost every game. If it loses, try training with more episodes."
        )


if __name__ == "__main__":
    main()
