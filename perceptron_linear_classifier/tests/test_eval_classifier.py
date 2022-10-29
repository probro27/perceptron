import sys
sys.path.append(".") # Adds higher directory to python modules path
import numpy as np
from perceptron_linear_classifier.perceptron import Perceptron

data1, labels1, data2, labels2 = \
(np.array([[-2.97797707,  2.84547604,  3.60537239, -1.72914799, -2.51139524,
         3.10363716,  2.13434789,  1.61328413,  2.10491257, -3.87099125,
         3.69972003, -0.23572183, -4.19729119, -3.51229538, -1.75975746,
        -4.93242615,  2.16880073, -4.34923279, -0.76154262,  3.04879591,
        -4.70503877,  0.25768309,  2.87336016,  3.11875861, -1.58542576,
        -1.00326657,  3.62331703, -4.97864369, -3.31037331, -1.16371314],
       [ 0.99951218, -3.69531043, -4.65329654,  2.01907382,  0.31689211,
         2.4843758 , -3.47935105, -4.31857472, -0.11863976,  0.34441625,
         0.77851176,  1.6403079 , -0.57558913, -3.62293005, -2.9638734 ,
        -2.80071438,  2.82523704,  2.07860509,  0.23992709,  4.790368  ,
        -2.33037832,  2.28365246, -1.27955206, -0.16325247,  2.75740801,
         4.48727808,  1.6663558 ,  2.34395397,  1.45874837, -4.80999977]]), 
 np.array([[-1., -1., -1., -1., -1., -1.,  1.,  1.,  1., -1., -1., -1., -1.,
        -1.,  1., -1.,  1., -1., -1., -1.,  1.,  1.,  1.,  1.,  1., -1.,
        -1., -1., -1., -1.]]), np.array([[ 0.6894022 , -4.34035772,  3.8811067 ,  4.29658177,  1.79692041,
         0.44275816, -3.12150658,  1.18263462, -1.25872232,  4.33582168,
         1.48141202,  1.71791177,  4.31573568,  1.69988085, -2.67875489,
        -2.44165649, -2.75008176, -4.19299345, -3.15999758,  2.24949368,
         4.98930636, -3.56829885, -2.79278501, -2.21547048,  2.4705776 ,
         4.80481986,  2.77995092,  1.95142828,  4.48454942, -4.22151738],
       [-2.89934727,  1.65478851,  2.99375325,  1.38341854, -4.66701003,
        -2.14807131, -4.14811829,  3.75270334,  4.54721208,  2.28412663,
        -4.74733482,  2.55610647,  3.91806508, -2.3478982 ,  4.31366925,
        -0.92428271, -0.84831235, -3.02079092,  4.85660032, -1.86705397,
        -3.20974025, -4.88505017,  3.01645974,  0.03879148, -0.31871427,
         2.79448951, -2.16504256, -3.91635569,  3.81750006,  4.40719702]]),
 np.array([[-1., -1.,  1.,  1., -1., -1., -1.,  1.,  1.,  1., -1.,  1.,  1.,
        -1.,  1.,  1.,  1., -1., -1., -1.,  1., -1.,  1., -1.,  1., -1.,
        -1.,  1.,  1.,  1.]]))

def test_eval_classifier():
    expected = 0.5333333333333333
    perceptron2 = Perceptron(data1, labels1, data2, labels2)
    perceptron2.fit()
    #print(data_train,labels_train)
    result = perceptron2.eval_classifier()
    assert(result==expected)