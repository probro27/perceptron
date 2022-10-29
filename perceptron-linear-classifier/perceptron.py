import numpy as np
from utils import y, cv, rv

class Perceptron:
    
    def __init__(self, data: np.ndarray, labels: np.ndarray, data_test: np.ndarray = None, labels_test: np.ndarray = None):
        # Dimensions
        d = data.shape[0]
        n = data.shape[1]
        
        # initializing variables
        self._data = data
        self._labels = labels
        self._theta = np.zeros(d).T
        self._theta0 = 0
        self._size = n
        self._current_step = 0
        self._data_test = data_test
        self._labels_test = labels_test
            
    def singleStep(self):
        x_i = self._data[:,self._current_step]
        y_i = self._labels[:,self._current_step]
        
        if y_i * y(x_i, self._theta, self._theta0):
            self._theta = self._theta + (y_i * x_i)
            self._theta0 = self._theta0 + y_i
        
        self._current_step += 1
    
    def singleIteration(self):
        for iteration in range(self._size):
            self.singleStep()
        self._current_step = 0
    
    def fit(self, T: int = 100):
        for t in range(T):
            self.singleIteration()
    
    def reset(self):
        self._theta = 0
        self._theta0 = 0
    
    def get_current_params(self):
        return {"theta": self._theta, "theta_0": self._theta0}
