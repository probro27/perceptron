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

    def eval_classifier(self):
        scored = score(self._data_test, self._labels_test, self._theta, self._theta0)
        return scored / self._size

    def xval_learning_alg(self, batch_size: int = 1):
        learner = self.fit()
        data = self._data
        labels = self._labels
        k = batch_size
        split_array = np.hsplit(data, data.shape[1])
        distributed_data = np.array_split(split_array, k)
        prev = 0
        scores = []
        for iteration in range(k):
            copy_dis_data = distributed_data
            data_i = np.array(copy_dis_data[iteration]).T[0]
            copy_dis_data_1 = copy_dis_data[:iteration] + copy_dis_data[iteration + 1:]
            data_j = np.concatenate(copy_dis_data_1, axis=0).T[0]
            labels_i = labels[:, prev:prev + data_i.shape[1]]
            selector = [x for x in range(labels.shape[1]) if x < prev or x >= prev + data_i.shape[1]]
            labels_j = labels[:, selector]
            prev += data_i.shape[1]
            (theta, theta_0) = learner(data_j, labels_j)
            scores.append(score(data_i, labels_i, theta, theta_0) / labels_i.shape[1])
        sum_score = 0
        for element in scores:
            sum_score += element
        return sum_score / k
