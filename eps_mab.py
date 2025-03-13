from typing import Optional
from mab import MAB, np


def get_best_index(qvalues: np.ndarray) -> int:
    """
    Sceglie il braccio con reward massima.
    Se ci sono più bracci con lo stesso valore massimo, ne seleziona uno."""
    indices = np.argwhere(qvalues == np.max(qvalues))  # Trova tutti gli indici con valore massimo
    idx = np.random.randint(0, len(indices))  # Sceglie casualmente tra questi
    return indices[idx][0]  # Restituisce l'indice scelto (il primo se ci sono più indici con lo stesso valore)


class EpsGreedyMAB(MAB):
    """
    Epsilon-Greedy multi-armed bandit
        :param n_arms : int [Number of arms (es. film da raccomandare)].
        :param n_dims: int [Numero di dimensioni del contesto (opzionale)]
        :param epsilon : float [Explore probability]
        :param Q0 : float [Initial value for the reward estimate of arms.]
    """

    def __init__(self, n_arms: int = 10, n_dims: Optional[int] = None, epsilon: float = 0.1, Q0: float = 0.0):
        super().__init__(n_arms, n_dims)
        if not (0 <= epsilon <= 1):
            raise ValueError("epsilon must be in [0,1]")
        if not isinstance(epsilon, float):
            raise TypeError("epsilon must be float")
        if not isinstance(Q0, float):
            raise TypeError("Q0 must be a float")
        self._epsilon = epsilon
        self._qvalues = np.full(n_arms, Q0)  # Per decidere quale braccio "sfruttare" (exploitation)
        self._rewards_list = np.zeros(n_arms)
        self._clicks = np.zeros(n_arms)

    def play(self, context: Optional[np.ndarray] = None) -> int:
        """Deve restituire un braccio (arm) da selezionare."""
        super().play(context)
        p = np.random.uniform(0, 1)
        if p <= self._epsilon:  #! Exploration
            # print(f"@Exploration con p: {p}")
            arm = np.random.randint(0, self._n_arms)
        else:  #! Exploitation
            arm = get_best_index(self._qvalues)
            # print(f"Exploitation con best arm: {arm}")
        return arm

    def update(self, arm: int, reward: float, context: Optional[np.ndarray] = None) -> None:
        """Aggiorna il valore del braccio selezionato con la ricompensa ottenuta."""
        super().update(arm, reward, context)
        self._clicks[arm] += 1
        self._rewards_list[arm] += reward
        self._qvalues[arm] = self._rewards_list[arm] / self._clicks[arm]

    def get_narms(self) -> int:
        return self._n_arms

    def get_ndim(self) -> int:
        return self._n_dims

    def get_qvalues(self) -> np.ndarray:
        return self._qvalues

    def get_last_reward(self):
        return self._reward

    def get_rewards_list(self) -> np.ndarray:
        return self._rewards_list

    def get_clicks_for_arm(self) -> int:
        return self._clicks

    def get_top_n(self) -> list[tuple[int, float]]:
        # Creiamo una lista di tuple (indice, Q-value)
        qvalues_with_indices = list(zip(range(self.get_narms()), self.get_qvalues()))
        # Ordiniamo la lista in base al Q-value
        qvalues_with_indices_sorted = sorted(qvalues_with_indices, key=lambda x: x[1], reverse=True)
        return qvalues_with_indices_sorted
