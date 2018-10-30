import numpy as np
from mnist import MNIST as mn
import os
import pickle
import numpy as np

def makeOneHot(y, numClasses = 1): #assuming 26 classes
    if(numClasses!=1):
        if(type(y)==np.ndarray):
            onehot = []
            for i in range(len(y)):
                temp = np.zeros((numClasses, ))
                temp[int(y[i])-1] = 1
                onehot.append(temp)
            c = np.array(onehot)
            c = np.transpose(c)
            return c
        else:
            y = int(y)
            onehot = np.zeros((numClasses,))
            onehot[y-1] = 1
            onehot = np.reshape(onehot, (numClasses,1))
            return onehot
    if(numClasses==1):
        return y

def extractSpecificSamples(labelsList):
    """given list of classes all training and testing samples of that class
       are returned
       also splits data into training and cross validation"""
    HOME = os.getcwd()
    mndata = mn(HOME+'/Total_Dataset')#mnist directory
    mndata.select_emnist('byclass')
    trainingImages, trainingLabels = mndata.load_training()
    testImages, testLabels = mndata.load_testing()
    selectedTrainImages = []
    selectedTrainLabels = []
    selectedTestImages = []
    selectedTestLabels = []
    for i in range(len(trainingLabels)):
        if(trainingLabels[i] in labelsList):
            selectedTrainImages.append(trainingImages[i])
            selectedTrainLabels.append(1+labelsList.index(trainingLabels[i]))
    for i in range(len(testLabels)):
        if(testLabels[i] in labelsList):
            selectedTestImages.append(testImages[i])
            selectedTestLabels.append(1+labelsList.index(testLabels[i]))
    os.chdir(HOME+'/SampledData')#target directory
    saveMatrix(testImages, 'testImages')
    saveMatrix(testLabels, 'testLabels')
    x = int(0.25*len(trainLabels))
    cvImages = trainImages[:x]
    saveMatrix(cvImages, 'cvImages')
    trainImages = trainImages[x:]
    saveMatrix(trainImages, 'trainImages')
    cvLabels = trainLabels[:x]
    saveMatrix(cvLabels, 'cvLabels')
    trainLabels = trainLabels[x:]
    saveMatrix(trainLabels, 'trainLabels')
    os.chdir(HOME)
    return None

def saveMatrix(matrix, filename):
    """function to save matrix in a pickle file"""
    with open(filename, 'wb') as _:
        pickle.dump(matrix, _)
    return None

def loadMatrix(filename):
    with open (filename, 'rb') as fp:
        matrix = np.array(pickle.load(fp))
    return matrix

def loadCSV(path):
    my_data = np.genfromtxt(path, delimiter=',')
    return my_data