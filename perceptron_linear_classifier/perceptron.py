import numpy as np

def cv(value_list):
    '''
    Takes a list of numbers and returns a column vector:  n x 1
    '''
    return np.transpose(rv(value_list))

def rv(value_list):
    '''
    Takes a list of numbers and returns a row vector: 1 x n
    '''
    return np.array([value_list])

def y(x, th, th0):
    '''
    x is dimension d by 1
    th is dimension d by 1
    th0 is a scalar
    return a 1 by 1 matrix
    '''
    return np.dot(np.transpose(th), x)[0] + th0

def positive(x, th, th0):
    '''
    x is dimension d by 1
    th is dimension d by 1
    th0 is dimension 1 by 1
    return 1 by 1 matrix of +1, 0, -1
    '''
    return np.sign(y(x, th, th0))

def score(data, labels, th, th0):
    '''
    data is dimension d by n
    labels is dimension 1 by n
    ths is dimension d by 1
    th0s is dimension 1 by 1
    return 1 by 1 matrix of integer indicating number of data points correct for
    each separator.
    '''
    return np.sum(positive(data, th, th0) == labels)


class Perceptron:
    
    def __init__(self, data: np.ndarray, labels: np.ndarray, data_test: np.ndarray = None, labels_test: np.ndarray = None):
        # Dimensions
        d = data.shape[0]
        n = data.shape[1]
        
        # initializing variables
        self._data = data
        self._labels = labels
        self._theta = np.zeros((d, 1))
        self._theta0 = np.zeros((1, 1))
        self._size = n
        self._current_step = 0
        self._data_test = data_test
        self._labels_test = labels_test
            
    def singleStep(self):
        x_i = self._data[:,self._current_step]
        y_i = self._labels[:,self._current_step]
        
        if y_i * y(x_i, self._theta, self._theta0) <= 0:
            self._theta = self._theta + np.array([(y_i * x_i)]).T
            self._theta0 = self._theta0 + y_i
        
        self._current_step += 1
    
    def singleIteration(self):
        for iteration in range(self._current_step, self._size):
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

    def eval_classifier(self):
        scored = score(self._data_test, self._labels_test, self._theta, self._theta0)
        return scored / self._size
