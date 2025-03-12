import numpy as np
from abc import ABC, abstractmethod

class MAB(ABC):
    """Base class for a contextual multi-armed bandit (MAB)
        :param n_arms: Numero di bracci
        :param n_dims: Numero di dimensioni del contesto (solo per bandit contestuali)
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

    @abstractmethod
    def play(self, context):
        """
        Deve restituire un braccio (arm) da selezionare.
            :param  context : [float numpy.ndarray, shape (n_arms, n_dims), optional]
             NOTE: Non-contextual bandits accept a context of None.
        """
        if context is not None:
            if not isinstance(context, np.ndarray):
                raise TypeError("context must be numpy.ndarray")
            if context.shape != (self._n_arms, self._n_dims):
                raise TypeError(f"context must have shape ({self._n_arms}, {self._n_dims})")
            self._context = context

    @abstractmethod
    def update(self, arm, reward, context):
        """
        Aggiorna il valore del braccio selezionato con la ricompensa ottenuta.
            :param arm : int index of the played arm in the set {0, ..., n_arms - 1}.
            :param reward : float [Reward received from the arm.]
            :param  context : [float numpy.ndarray, shape (n_arms, n_dims), optional]
             NOTE: Non-contextual bandits accept a context of None.
        """
        if context is not None:
            if not isinstance(context, np.ndarray):
                raise TypeError("context must be numpy.ndarray")
            if context.shape != (self._n_arms, self._n_dims):
                raise TypeError(f"context must have shape ({self._n_arms}, {self._n_dims})")
        self._arm = arm
        self._reward = reward
        self._context = context