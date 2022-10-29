import sys
sys.path.append(".") # Adds higher directory to python modules path
from perceptron_linear_classifier.perceptron import Perceptron
import numpy as np

def super_simple_separable_through_origin():
    '''
    Return d = 2 by n = 4 data matrix and 1 x n = 4 label matrix
    '''
    X = np.array([[2, 3, 9, 12],
                  [5, 1, 6, 5]])
    y = np.array([[1, -1, 1, -1]])
    return X, y

def xor():
    '''
    Return d = 2 by n = 4 data matrix and 1 x n = 4 label matrix
    '''
    X = np.array([[1, 2, 1, 2],
                  [1, 2, 2, 1]])
    y = np.array([[1, 1, -1, -1]])
    return X, y

expected_perceptron = [(np.array([[-9.0], [18.0]]), np.array([[2.0]])),(np.array([[0.0], [-3.0]]), np.array([[0.0]]))]
datasets = [super_simple_separable_through_origin,xor]

def incorrect(expected,result):
    print("Test Failed.")
    print("Your code output ",result)
    print("Expected ",expected)
    print("\n")

def correct():
    print("Passed! \n")


def test_perceptron():
    '''
    Checks perceptron theta and theta0 values for 100 iterations
    '''
    for index in range(len(datasets)):
        data, labels = datasets[index]()
        perceptron = Perceptron(data, labels)
        perceptron.fit(100)
        params = perceptron.get_current_params()
        th = params['theta']
        th0 = params['theta_0']
        expected_th,expected_th0 = expected_perceptron[index]
        print("-----------Test Perceptron "+str(index)+"-----------")
        if((th==expected_th).all() and (th0==expected_th0).all()):
            correct()
        else:
            incorrect("th: "+str(expected_th.tolist())+", th0: "+str(expected_th0.tolist()), "th: "+str(th.tolist())+", th0: "+str(th0.tolist()))
        assert((th==expected_th).all() and (th0==expected_th0).all())
