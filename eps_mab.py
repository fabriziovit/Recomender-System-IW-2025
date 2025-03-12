from mab import MAB, np
import pandas as pd

def get_index_of_best(qvalues: np.ndarray) -> int:
    """ 
    Sceglie il braccio con rewar massima.
    Inoltre, gestisce il caso in cui più bracci abbiano lo stesso valore massimo."""
    indices = np.argwhere(qvalues == np.max(qvalues))  # Trova tutti gli indici con valore massimo
    index = np.random.randint(0, len(indices)) # Sceglie casualmente tra questi
    return indices[index][0] # Restituisce l'indice scelto

class EpsGreedy(MAB):
    """
    Epsilon-Greedy multi-armed bandit
        :param n_arms : int [Number of arms (es. film da raccomandare)].
        :param n_dims: int [Numero di dimensioni del contesto (solo per bandit contestuali)]
        :param epsilon : float [Explore probability (Probabilità di esplorare scelte nuove.)]
        :param Q0 : float, default=np.inf [Initial value for the reward of arms. (Valore iniziale per la stima della ricompensa di ogni braccio)]
    """
    def __init__(self, n_dims: int, n_arms: int =10, epsilon: float =0.1, Q0: float =np.inf):
        super().__init__(n_arms, n_dims)
        if not (0 <= epsilon <= 1):
            raise ValueError("epsilon must be in [0,1]")
        if not isinstance(epsilon, float):
            raise TypeError("epsilon must be float")
        if not isinstance(Q0, float):
            raise TypeError("Q0 must be a float")
        self._epsilon = epsilon
        self._qvalues = np.full(n_arms, Q0) #Per decidere quale braccio "sfruttare" (exploitation)
        self._rewards = np.zeros(n_arms)
        self._clicks = np.zeros(n_arms)

    def play(self, context : np.ndarray = None) -> int:
        """ Deve restituire un braccio (arm) da selezionare. """
        super().play(context)
        p = np.random.uniform(0,1)
        if p <= self._epsilon: #Exploration 
            #print(f"@Exploration con p: {p}")  
            arm = np.random.randint(0, self._n_arms)
        else: #Exploitation 
            arm = get_index_of_best(self._qvalues)  
            #print(f"Exploitation con best arm: {arm}") 
        return arm

    def update(self, arm: int, reward: float, context: np.ndarray = None) -> None:
        """ Aggiorna il valore del braccio selezionato con la ricompensa ottenuta. """
        super().update(arm, reward, context)
        self._clicks[arm] += 1
        self._rewards[arm] += reward 
        self._qvalues[arm] = self._rewards[arm] / self._clicks[arm]

    def get_narms(self) -> int:
        return self._n_arms
    
    def get_ndim(self) -> int:
        return self._n_dims
    
    def get_qvalues(self) -> np.ndarray:
        return self._qvalues
    
    def get_reward(self) -> np.ndarray:
        return self._reward
    
    def get_clicks(self) -> int:
        return self._clicks
    
    
    def get_top_n(self, n_top=5) -> list[tuple[int, float]]:
        # Creiamo una lista di tuple (indice, Q-value)
        qvalues_with_indices = list(zip(range(self.get_narms()), self.get_qvalues()))
        # Ordiniamo in base al Q-value in ordine decrescente
        qvalues_with_indices_sorted = sorted(qvalues_with_indices, key=lambda x:x[1], reverse=True)

        list_top_n_tuple = qvalues_with_indices_sorted[0:n_top] # Prendiamo solo i primi n_top

        return list_top_n_tuple