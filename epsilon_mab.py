from typing import Optional
import numpy as np
from abc import ABC, abstractmethod


# *** Base MAB class ***#
class MAB(ABC):
    """Base class for a contextual Multi-Armed Bandit (MAB)
    n_arms: Number of arms
    n_dims: Number of context dimensions (only for contextual bandits)"""

    def __init__(self, n_arms: int, n_dims: Optional[int] = None):
        if not isinstance(n_arms, int):
            raise TypeError("n_arms must be an integer")
        if n_arms < 0:
            raise ValueError("n_arms must be non-negative")
        if n_dims is not None:
            if not isinstance(n_dims, int):
                raise TypeError("n_dims must be an integer")
            if n_dims < 0:
                raise ValueError("n_dims must be non-negative")
            self._n_dims = n_dims
        self._n_arms = n_arms

    def _validate_context(self, context: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        if context is not None:
            if not isinstance(context, np.ndarray):
                raise TypeError("context must be numpy.ndarray")
            if context.shape != (self._n_arms, self._n_dims):
                raise TypeError(f"context must have shape ({self._n_arms}, {self._n_dims})")
        return context

    def _validate_arm(self, arm: int) -> int:
        if not isinstance(arm, (int, np.integer)):
            raise TypeError("arm must be an integer")
        if arm < 0 or arm >= self._n_arms:
            raise ValueError("arm must be between 0 and n_arms-1")
        return arm

    def _validate_reward(self, reward: float) -> float:
        if not isinstance(reward, (float, np.floating)):
            raise TypeError("reward must be a number")
        return reward

    @abstractmethod
    def play(self, context: Optional[np.ndarray] = None) -> int:
        """Must return an arm to select."""
        self._context = self._validate_context(context)

    @abstractmethod
    def update(self, arm: int, reward: float, context: Optional[float] = None) -> None:
        """Updates the value of the selected arm with the obtained reward."""
        self._arm = self._validate_arm(arm)
        self._reward = self._validate_reward(reward)
        self._context = self._validate_context(context)


def get_best_arm(qvalues: np.ndarray) -> int:
    """Chooses the arm with maximum reward.
    If there are multiple arms with the same maximum value, selects one of them."""
    indices = np.argwhere(qvalues == np.max(qvalues))  # Find all indices with maximum value
    idx = np.random.randint(0, len(indices))  # Randomly choose among these
    return indices[idx][0]  # Return the chosen index (the first if there are multiple indices with the same value)


# *** Epsilon-Greedy MAB ****#
class EpsGreedyMAB(MAB):
    """Epsilon-Greedy multi-armed bandit
    :param n_arms : int [Number of arms (e.g. movies to recommend)].
    :param n_dims: int [Number of context dimensions (optional)]
    :param epsilon : float [Explore probability]
    :param Q0 : float [Initial value for the reward estimate of arms.]
    """

    def __init__(self, n_arms: int = 10, n_dims: Optional[int] = None, epsilon: float = 0.1, Q0: float = 0.0):
        super().__init__(n_arms, n_dims)
        if not (0.0 <= epsilon <= 1.0):
            raise ValueError("epsilon must be in [0,1]")
        if not isinstance(epsilon, float):
            raise TypeError("epsilon must be float")
        if not isinstance(Q0, float):
            raise TypeError("Q0 must be a float")
        self._initial_epsilon = epsilon
        self._curr_epsilon = epsilon
        self._epsilon_decay_function = None  # Epsilon decay function
        self._qvalues = np.full(n_arms, Q0)  # To decide which arm to "exploit" (exploitation)
        self._total_rewards = np.zeros(n_arms)
        self._clicks = np.zeros(n_arms)
        self._nexploitation = 0  # Number of exploitations performed
        self._nexploration = 0  # Number of explorations performed

    def play(self, context: Optional[np.ndarray] = None) -> int:
        """Must return an arm to select."""
        super().play(context)
        p = np.random.uniform(0, 1)
        if p <= self._curr_epsilon:  # Exploration
            self._nexploration += 1
            arm = np.random.randint(0, self._n_arms)
        else:  # Exploitation
            self._nexploitation += 1
            arm = get_best_arm(self._qvalues)
        return arm

    def update(self, arm: int, reward: float, context: Optional[np.ndarray] = None) -> None:
        """Updates the value of the selected arm with the obtained reward."""
        super().update(arm, reward, context)
        self._clicks[arm] += 1
        self._total_rewards[arm] += reward
        # Maintains the average of observed rewards for each arm.
        self._qvalues[arm] = self._total_rewards[arm] / self._clicks[arm]

    def get_narms(self) -> int:
        return self._n_arms

    def get_ndim(self) -> int:
        return self._n_dims

    def get_qvalues(self) -> np.ndarray:
        return self._qvalues

    def get_total_rewards_list(self) -> np.ndarray:
        return self._total_rewards

    def get_clicks_for_arm(self) -> np.ndarray:
        return self._clicks

    def get_curr_epsilon(self) -> float:
        return self._curr_epsilon

    def get_top_n(self) -> list[tuple[int, float]]:
        # Create a list of tuples (index, Q-value)
        qvalues_with_indices = list(zip(range(self.get_narms()), self.get_qvalues()))
        # Sort the list based on Q-value
        qvalues_with_indices_sorted = sorted(qvalues_with_indices, key=lambda x: x[1], reverse=True)
        return qvalues_with_indices_sorted

    def set_epsilon_deacy(self, function: callable) -> None:
        """Sets the epsilon decay function."""
        if not callable(function):
            raise TypeError("function must be callable")
        self._epsilon_decay_function = function

    def update_epsilon(self, num_round) -> None:
        if not hasattr(self, "_epsilon_decay_function"):
            raise ValueError("epsilon decay function not set")
        new_epsilon = self._epsilon_decay_function(self._initial_epsilon, num_round)
        if not isinstance(new_epsilon, (float, np.floating)):
            raise TypeError("epsilon must be float")
        # Ensure epsilon is between 0 and 1 (Clamping)
        new_epsilon = max(0.0, min(1.0, new_epsilon))
        self._curr_epsilon = new_epsilon
