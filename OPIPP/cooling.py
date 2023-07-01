#coding:UTF-8

""" Cooling Schedules """

import numpy as np

MIN_T = 1e-4
MAX_UPDATE = None 

class Schedule:
    def __init__(self, init_t: float, min_t: float, max_update: int=MAX_UPDATE) -> None:
        self.init_t = init_t
        self.min_t = min_t
        self.max_update = max_update

    def init(self) -> None:
        self.t = self.init_t
        self.i_update = 0

    def next(self, loss: float) -> float:
        self.i_update += 1
        self.update(loss)
        return self.t

    def update(self, loss) -> None:
        pass

    def has_next(self) -> bool:
        if self.max_update is None:
            return self.t > self.min_t
        else:
            return self.t > self.min_t and self.i_update < self.max_update

class AdaptiveSchedule(Schedule):
    def __init__(self, alpha: float=0.95, init_t: float=0.5, min_t: float=MIN_T):
        self.alpha = alpha
        Schedule.__init__(self, init_t=init_t, min_t=min_t)

    def init(self) -> None:
        self.mean = 0
        Schedule.init(self)

    def update(self, loss) -> None:
        now_mean = (self.mean*(self.i_update-1)+loss)/self.i_update
        previous_mean = self.mean
        self.mean = now_mean
        if now_mean > previous_mean:
            self.t *= self.alpha


