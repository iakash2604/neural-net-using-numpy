import os
import numpy as np
# from mnist import MNIST as mn
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from neuralNetwork import *
from utils import *
import functions as fns

print('loading data')
# trainImages = np.array(loadMatrix('./SampledData/trainImages'))
# trainLabels = np.array(loadMatrix('./SampledData/trainLabels'))
# cvImages = np.array(loadMatrix('./SampledData/cvImages'))
# cvLabels = np.array(loadMatrix('./SampledData/cvLabels'))
# trainImages = loadCSV('./toy_data/toy_trainX.csv')
# trainLabels = loadCSV('./toy_data/toy_trainY.csv')
# cvImages = loadCSV('./toy_data/toy_testX.csv')
# cvLabels = loadCSV('./toy_data/toy_testY.csv')
print('loaded data')

nn = neuralNetwork(name="demo",
                   epochs=2,
                   batchSize=32,
                   alpha=0.05,
                   L2=0.0,
                   L1=0,
                   layerSizes=[784, 100, 10],
                   actFn=fns.identity,
                   costFn=fns.meanSquare,
                   lastLayerFn=fns.identity,
                   dropout=0,
                   batchNorm=False)

nn.makeWeightAndBiasMatrices()
graphs = nn.miniBatchGradientDescent(trainImages, trainLabels, cvImages, cvLabels)
nn.saveModel()
