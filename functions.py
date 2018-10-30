import numpy as np
from utils import *
# np.seterr(divide='ignore', invalid='ignore')


class sigmoid:
    def activation(y):
        return (1/(1+np.exp(-y)))

    def derivative(y):
        return sigmoid.activation(y)*(1-sigmoid.activation(y))

class ReLU:
    def activation(y):
        return np.multiply((y>0).astype(np.int), y)

    def derivative(y):
        return (y>0).astype(np.int)

class tanh:
    def activation(y):
        return (np.exp(y)-np.exp(-y))/(np.exp(y)+np.exp(-y))

    def derivative(y):
        return 1-(tanh.activation(y))**2

class identity:
    def activation(y):
        return y

    def derivative(y):
        return 1

class softmax:
    def activation(y):
        exps = np.exp(y-np.max(y, axis=0))
        return exps/np.sum(exps, axis=0)

    def derivative(y):
        return np.multiply(softmaxActivation(y), (1-softmaxActivation(y)))

#########################################################

class meanSquare:
    def cost(y, y_):
        return 0.5*np.linalg.norm(y-y_)**2

    def derivative(y, y_, z):
        y_ = makeOneHot(y_)
        c = (y-y_)*sigmoid.derivative(z)
        return c

class crossEntropy:
    def cost(y, y_):
        y_ = makeOneHot(y_)
        return np.sum(np.nan_to_num(-y_*np.log(y)-(1-y_)*np.log(1-y)))

    def derivative(y, y_, z):
        y_ = makeOneHot(y_)
        return (y-y_)
