import numpy as np

class Distribution:
    def __init__(self, max_value: float, n_bin: int, target_probs: list, min_value: float=0.):
        assert len(target_probs) == n_bin+1
        assert max_value > min_value
        self.max_value = max_value
        self.min_value = min_value
        self.n_bin = n_bin
        self.target_probs = target_probs
        # make sure the correct calculation for KL distance
        self.target_probs[self.target_probs<=0] = 1e-5 
        self.target_probs /= self.target_probs.sum()
        self.step = (max_value-min_value)/n_bin

    def __get_index(self, value: float) -> int:
        return min(int((value-self.min_value)/self.step), self.n_bin)

    def get_prob(self, value: float) -> float:
        return self.target_probs[self.__get_index(value)]

    def get_value_centers(self) -> np.ndarray:
        return np.linspace(self.min_value, self.max_value, num=self.n_bin+1)+self.step/2

    def get_hist(self, values: np.ndarray) -> np.ndarray:
        hist, _ = np.histogram(values, bins=self.n_bin, range=(self.min_value, self.max_value))
        hist = np.concatenate((hist, [len(values)-sum(hist)])).astype(float)
        return hist

    def KL(self, distances: np.ndarray) -> float:
        hist = self.get_hist(distances)
        probs = hist/hist.sum()
        kl = np.sum([pk * np.log(pk / qk) for pk, qk in zip(probs, self.target_probs) if qk > 0 and pk > 0])
        return kl

    def sample_values(self, n: int=1) -> np.ndarray:
        centers = np.random.choice(self.get_value_centers(), p=self.target_probs, size=n)
        return (np.random.rand(n)-0.5)*self.step+centers

    def sample_ri(self, n_sample=50000) -> float:
        values = self.sample_values(n_sample)
        return values.mean()/values.std()



