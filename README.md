# Perceptron-Linear-Classifier
A Python library to implement the perceptron algorithm and possibly visualize it. 

## Usage

## Examples of How To Use 

Creating An Object

```python
from perceptron_linear_classifier import Perceptron

perceptron = Perceptron(data=data_train, labels=labels_train, data_test=data_test, labels_test=labels_test)

# train the perceptron classifier to find the respective values of theta and theta_0
perceptron.fit()

# returns the score for the perceptron classifier
perceptron.eval_classifier()

# perform 1 step of the perceptron classifier to check if the value of theta and theta0 changes
perceptron.singleStep()

# perform 1 iteration of passing the entire data to the classifier 
perceptron.singleIteration()
```

## Contributing

Please read the CONTRIBUTING.md file for more information about it. 

## Issues

Report to Prabhav Khera. 
