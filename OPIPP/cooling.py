import numpy as np

MIN_T = 1e-4
MAX_UPDATE = None 

class Schedule:
    """
    Cooling Schedule for the SA optimization
    
    Methods
    -------
    init()
        Initalization.

    update()
        Update inner states with a new entropy.

    has_next()
        True if not terminate.
        
    Usage
    -------
    Please follow the routine as
    // create a object
    schedule = Schedule()
    // init its inner states
    schedule.init()
    foreach iteration:
        generate a new entropy
        // update
        schedule.update(entropy)
        //check if it is terminated
        if not schedule.has_next():
            break
    """
    def __init__(self, min_t: float, max_update: int=MAX_UPDATE) -> None:
        """
        Args:
            min_t (float): Threshold for termination.
            max_update (int, optional): Maximum number of iteration. None for no limitation. Defaults to None.
        """        
        self.min_t = min_t
        self.max_update = max_update

    def init(self) -> None:
        """initialize inner states
        """        
        self.i_update = 0

    def next(self, loss: float) -> float:
        """forward step

        Args:
            loss (float): new entropy

        Returns:
            float: temperature
        """        
        self.i_update += 1
        self.update(loss)
        return self.t

    def update(self, loss: float) -> None:
        """update inner temperature
        Please rewrite this method in children classes

        Args:
            loss (float): new entropy
        """        
        pass

    def has_next(self) -> bool:
        """check if the cooling schedule is terminated

        Returns:
            bool: True if it can recieve new entropy
        """        
        if self.max_update is None:
            return self.t > self.min_t
        else:
            return self.t > self.min_t and self.i_update < self.max_update

class AdaptiveSchedule(Schedule):
    """ 
    the adaptive cooling schedule for the SA optimization
    """
    def __init__(self, alpha: float=0.95, init_t: float=0.5, min_t: float=MIN_T):
        """
        Args:
            alpha (float, optional): Control the speed of adaptation. Defaults to 0.95.
            init_t (float, optional): The value of temperature at initalization. Defaults to 0.5.
            min_t (float, optional): The value of temperature for termination. Defaults to MIN_T=1e-4.
        """        
        self.alpha = alpha
        self.init_t = init_t
        Schedule.__init__(self, min_t=min_t)

    def init(self) -> None:
        self.mean = 0 # record the average of previous iterations
        self.t = self.init_t # current t
        Schedule.init(self)

    def update(self, loss) -> None:
        now_mean = (self.mean*(self.i_update-1)+loss)/self.i_update
        previous_mean = self.mean
        self.mean = now_mean
        if now_mean > previous_mean:
            self.t *= self.alpha


