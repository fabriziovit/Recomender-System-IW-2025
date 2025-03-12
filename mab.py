import numpy as np
from abc import ABC, abstractmethod


class MAB(ABC):
    """
    Base class for a contextual Multi-Armed Bandit (MAB)
    n_arms: Numero di bracci
    n_dims: Numero di dimensioni del contesto (solo per bandit contestuali)
    """

    def __init__(self, n_arms, n_dims):
        if not isinstance(n_arms, int):
            raise TypeError("n_arms must be an integer")
        if n_arms < 0:
            raise ValueError("n_arms must be non-negative")
        if not isinstance(n_dims, int):
            raise TypeError("n_dims must be an integer")
        if n_dims < 0:
            raise ValueError("n_dims must be non-negative")
        self._n_arms = n_arms
        self._n_dims = n_dims

    def _validate_context(self, context) -> np.ndarray:
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
        if not isinstance(reward, (int, (float, np.floating))):
            raise TypeError("reward must be a number")
        return float(reward)

    @abstractmethod
    def play(self, context):
        """Deve restituire un braccio (arm) da selezionare."""
        self._context = self._validate_context(context)

    @abstractmethod
    def update(self, arm, reward, context):
        """Aggiorna il valore del braccio selezionato con la ricompensa ottenuta."""
        self._arm = self._validate_arm(arm)
        self._reward = self._validate_reward(reward)
        self._context = self._validate_context(context)
