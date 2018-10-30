import numpy as np
from utils import *
import functions as fns
import pickle
from matplotlib import pyplot as plt

class neuralNetwork:
    def __init__(self, name, epochs, batchSize, alpha, L2, L1, layerSizes, actFn, costFn, lastLayerFn, dropout, batchNorm):
        self.name = name
        self.layerSizes = layerSizes
        self.numHidden = len(layerSizes)-2
        self.epochs = epochs
        self.batchSize = batchSize
        self.alpha = alpha
        self.L1 = L1
        self.L2 = L2
        self.actFn = actFn
        self.costFn = costFn
        self.lastLayerFn = lastLayerFn
        self.weightMatrices = [None]*(self.numHidden+1)
        self.biasMatrices = [None]*(self.numHidden+1)
        self.trackCostsCV = []
        self.trackCostsTrain = []
        self.trackAccuracyCV = []
        self.trackAccuracyTrain = []
        self.dropProb = 1-dropout
        self.dropoutNetwork = [None]*(self.numHidden+1)
        self.batchNorm = batchNorm
        self.eps=1e-5
        if self.batchNorm:
            self.gammasMatrices=[None]*(self.numHidden)
            self.betasMatrices=[None]*(self.numHidden)

    def makeWeightAndBiasMatrices(self):
        """initialize weights and biases for network"""
        for i in range(self.numHidden+1):
            self.weightMatrices[i] = np.random.randn(self.layerSizes[i+1], self.layerSizes[i])/np.sqrt(self.layerSizes[i+1])
            self.biasMatrices[i] = np.random.randn(self.layerSizes[i+1], 1)
        if self.batchNorm:
            for k in range(self.numHidden):
                self.gammasMatrices[k] = np.ones((self.layerSizes[k+1],1))
                self.betasMatrices[k] = np.zeros((self.layerSizes[k+1],1))
        return None

    def outputNeural(self, x):
        """computes the output layer values for performance testing of current
            state of the network"""
        x = x/np.max(x)
        x = np.transpose(x)
        for i in range(self.numHidden):
            x = np.dot(self.weightMatrices[i], x) + self.biasMatrices[i]
            if(self.batchNorm):
                x = (x-np.mean(x,axis=0))/(np.sqrt(np.var(x,axis=0)+self.eps))
                x = x*self.gammasMatrices[i] + self.betasMatrices[i]
            x = self.actFn.activation(x)
        x = np.dot(self.weightMatrices[self.numHidden], x) + self.biasMatrices[self.numHidden]
        y = self.lastLayerFn.activation(x)
        return y

    def forwardPass(self, x):
        """stores the preactivation (wx+b) and post activation (f(wx+b)) for
            neurons of all layers. used for calculating gradients using backprop"""
        x = x/np.max(x)
        preActivationLayers = [None]*(self.numHidden+1)
        postActivationLayers = [None]*(self.numHidden+2)
        postActivationLayers[0] = x
        for i in range(self.numHidden):
            x = np.dot(self.weightMatrices[i]*self.dropoutNetwork[i], x) + self.biasMatrices[i]
            preActivationLayers[i] = x
            x = self.actFn.activation(x)
            postActivationLayers[i+1] = x
        x = np.dot(self.weightMatrices[self.numHidden]*self.dropoutNetwork[self.numHidden], x) + self.biasMatrices[self.numHidden]
        preActivationLayers[self.numHidden] = x
        y = self.lastLayerFn.activation(x)
        postActivationLayers[self.numHidden+1] = y
        return [preActivationLayers, postActivationLayers]

    def forwardPassBatchNorm(self,x):
        x = x/np.max(x)
        preActivationLayers = [None]*(self.numHidden+1)
        postActivationLayers = [None]*(self.numHidden+2)
        normActivationLayers = [None]*(self.numHidden)
        postActivationLayers[0] = x
        for i in range(self.numHidden):
            x = np.dot(self.weightMatrices[i]*self.dropoutNetwork[i], x) + self.biasMatrices[i]
            normActivationLayers[i]=(x-np.mean(x,axis=0))/(np.sqrt(np.var(x,axis=0)+self.eps))
            x=self.gammasMatrices[i]*normActivationLayers[i]+self.betasMatrices[i]
            preActivationLayers[i] = x
            x = self.actFn.activation(x)
            postActivationLayers[i+1] = x
        x = np.dot(self.weightMatrices[self.numHidden]*self.dropoutNetwork[self.numHidden], x) + self.biasMatrices[self.numHidden]
        preActivationLayers[self.numHidden] = x
        y = self.lastLayerFn.activation(x)
        postActivationLayers[self.numHidden+1] = y
        return [preActivationLayers, postActivationLayers,normActivationLayers]

    def makeDropoutNetwork(self):
        for i in range(self.numHidden+1):
            shape = self.weightMatrices[i].shape
            probs = np.random.binomial(size=shape[1], n=1, p= self.dropProb)
            drop = np.zeros(shape=shape)
            for k in range(shape[1]):
                drop[:,k] = np.ones(shape=shape[0])*probs[k]
            self.dropoutNetwork[i] = drop
        return None

    def backProp(self, x, y_):
        """stochastic backprop, x & y_ should be single sample, not batch
            returns the gradients dC/dW and dC/dB"""
        weightMatricesUpdate = [None]*(self.numHidden+1)
        biasMatricesUpdate = [None]*(self.numHidden+1)
        for i in range(self.numHidden+1):
            biasMatricesUpdate[i] = np.zeros(shape=self.biasMatrices[i].shape)
            weightMatricesUpdate[i] = np.zeros(shape=self.weightMatrices[i].shape)
        vals = self.forwardPass(x) #activation values
        preActivationLayers = vals[0]
        postActivationLayers = vals[1]
        delta = self.costFn.derivative(postActivationLayers[-1], y_, preActivationLayers[-1])
        biasMatricesUpdate[-1] = delta
        weightMatricesUpdate[-1] = np.dot(delta, postActivationLayers[-2].transpose())*self.dropoutNetwork[-1]
        for l in range(2, self.numHidden+2):
            z = preActivationLayers[-l]
            sp = self.actFn.derivative(z)
            delta = np.dot(self.weightMatrices[-l+1].transpose(), delta) * sp
            biasMatricesUpdate[-l] = delta
            weightMatricesUpdate[-l] = np.dot(delta, postActivationLayers[-l-1].transpose())*self.dropoutNetwork[-l]
        return [weightMatricesUpdate, biasMatricesUpdate]

    def backPropBatchNorm(self,x,y_):
        weightMatricesUpdate = [None]*(self.numHidden+1)
        biasMatricesUpdate = [None]*(self.numHidden+1)
        gammasMatricesUpdate = [None]*(self.numHidden)
        betasMatricesUpdate = [None]*(self.numHidden)
        for i in range(self.numHidden+1):
            biasMatricesUpdate[i] = np.zeros(shape=self.biasMatrices[i].shape)
            weightMatricesUpdate[i] = np.zeros(shape=self.weightMatrices[i].shape)
        for i in range(self.numHidden):
            gammasMatricesUpdate[i] = np.zeros(shape=self.gammasMatrices[i].shape)
            betasMatricesUpdate[i] = np.zeros(shape=self.betasMatrices[i].shape)
        vals=self.forwardPassBatchNorm(x)
        preActivationLayers = vals[0]
        postActivationLayers = vals[1]
        normActivationLayers = vals[2]

        delta = self.costFn.derivative(postActivationLayers[-1], y_, preActivationLayers[-1])
        biasMatricesUpdate[-1] = delta
        weightMatricesUpdate[-1] = np.dot(delta, postActivationLayers[-2].transpose())*self.dropoutNetwork[-1]

        for l in range(2, self.numHidden+2):
            z = preActivationLayers[-l]
            sp = self.actFn.derivative(z)
            #dZ~
            delta = np.dot(self.weightMatrices[-l+1].transpose(), delta) * sp

            biasMatricesUpdate[-l] = delta
            betasMatricesUpdate[-l+1]= delta
            weightMatricesUpdate[-l] = np.dot(delta, postActivationLayers[-l-1].transpose())*self.dropoutNetwork[-l]
            gammasMatricesUpdate[-l+1]= normActivationLayers[-l+1]*delta
        return [weightMatricesUpdate, biasMatricesUpdate,gammasMatricesUpdate,betasMatricesUpdate]

    def miniBatchGradientDescent(self, x, y_, cvx, cvy_):
        """creates batches, finds corresponding gradient and updates weights and biases"""
        t = len(x)
        for epoch in range(self.epochs):
            print('epoch: ', epoch)
            print('training')
            cx = np.copy(x)
            cy_ = np.copy(y_)
            batches = int(t/self.batchSize)
            for j in range(batches):
                print(epoch, batches-j-1)
                batchImage = cx[:self.batchSize]
                batchLabel = cy_[:self.batchSize]
                cx = cx[self.batchSize:]
                cy_ = cy_[self.batchSize:]
                self.makeDropoutNetwork()
                self.updateRule(batchImage, batchLabel, t)
            print('cross-validation')
            valsCV = self.getAccuracyAndCost(cvx, cvy_)
            valsT = self.getAccuracyAndCost(x, y_)
            print('costCV: ', valsCV[1])
            print('accuracyCV: ', valsCV[0])
            print('costT: ', valsT[1])
            print('accuracyT: ', valsT[0])
            self.trackCostsCV.append(valsCV[1])
            self.trackAccuracyCV.append(valsCV[0])
            self.trackCostsTrain.append(valsT[1])
            self.trackAccuracyTrain.append(valsT[0])
        x = np.arange(self.epochs)+1
        ta = self.trackAccuracyTrain
        tc = self.trackCostsTrain
        cva = self.trackAccuracyCV
        cvc = self.trackCostsCV
        fig, ax = plt.subplots()
        plt.subplot(2, 1, 1)
        plt.plot(x, ta, 'g-', x, cva, 'b-')
        plt.title('Performance')
        plt.ylabel('Accuracy')
        plt.subplot(2, 1, 2)
        plt.plot(x, tc, 'g-', x, cvc, 'b-')
        plt.xlabel('epochs')
        plt.ylabel('Cost')

        plt.show()
        return None

    def updateRule(self, batchImage, batchLabel, t):
        """finds sum of all gradients in a batch and updates using gradient descent"""
        biasMatricesUpdate = [None]*(self.numHidden+1)
        weightMatricesUpdate = [None]*(self.numHidden+1)
        for i in range(self.numHidden+1):
            biasMatricesUpdate[i] = np.zeros(shape=self.biasMatrices[i].shape)
            weightMatricesUpdate[i] = np.zeros(shape=self.weightMatrices[i].shape)
        if(self.batchNorm):
            betasMatricesUpdate = [None]*(self.numHidden)
            gammasMatricesUpdate = [None]*(self.numHidden)
            for i in range(self.numHidden):
                betasMatricesUpdate = np.zeros(shape = self.betasMatrices[i].shape)
                gammasMatricesUpdate = np.zeros(shape = self.gammasMatrices[i].shape)

        for x, y_ in zip(batchImage, batchLabel): #sum over gradients for all samples in a batch
            x = np.reshape(x, (self.layerSizes[0], 1))
            if(self.batchNorm):
                updates = self.backPropBatchNorm(x, y_)
                weightUpdate = updates[0]
                biasUpdate = updates[1]
                gammasUpdate = updates[2]
                betasUpdate = updates[3]
                biasMatricesUpdate = [bMU+bU for bMU, bU in zip(biasMatricesUpdate, biasUpdate)]
                weightMatricesUpdate = [wMU+wU for wMU, wU in zip(weightMatricesUpdate, weightUpdate)]
                gammasMatricesUpdate = [gMU+gU for gMU, gU in zip(gammasMatricesUpdate, gammasUpdate)]
                betasMatricesUpdate = [bmu+bu for bmu, bu in zip(betasMatricesUpdate, betasUpdate)]
            else:
                updates = self.backProp(x, y_)
                weightUpdate = updates[0]
                biasUpdate = updates[1]
                biasMatricesUpdate = [bMU+bU for bMU, bU in zip(biasMatricesUpdate, biasUpdate)]
                weightMatricesUpdate = [wMU+wU for wMU, wU in zip(weightMatricesUpdate, weightUpdate)]

        self.weightMatrices = [w-np.sign(w)*self.L1 for w in self.weightMatrices] #L1 regularization step
        self.weightMatrices = [(1-self.alpha*(self.L2/t))*w-(self.alpha/len(batchImage))*wMU for w, wMU in zip(self.weightMatrices, weightMatricesUpdate)] #L2 regularization
        self.biasMatrices = [b-(self.alpha/len(batchImage))*bMU for b, bMU in zip(self.biasMatrices, biasMatricesUpdate)]
        if(self.batchNorm):
            self.gammasMatrices = [x-(self.alpha)/(len(batchImage))*y for x, y in zip(self.gammasMatrices, gammasMatricesUpdate)]
            self.betasMatrices = [x-(self.alpha)/(len(batchImage))*y for x, y in zip(self.betasMatrices, betasMatricesUpdate)]
        return None

    def getAccuracyAndCost(self, x, y_):
        """testing model performance"""
        accuracy = 0
        cost = 0
        t = x.shape[0]
        y = self.outputNeural(x)
        if(self.layerSizes[-1] == 1):
            accuracy = (y==y_).astype(int).sum()
            cost = self.costFn.cost(y, y_)
        else:
            for j in range(y.shape[1]):
                if(np.argmax(y[:, j]) == y_[j]-1):
                    accuracy = accuracy+1
            cost = cost + (self.costFn.cost(y, y_))
        return [accuracy/t, cost/t]

    def saveModel(self):
        file = open(self.name+"expLogs.txt","a")
        file.write("Layersizes: "+ str(self.layerSizes))
        file.write("\n")
        file.write("Batch size: "+ str(self.batchSize))
        file.write("\n")
        file.write("Epochs: "+ str(self.epochs))
        file.write("\n")
        file.write("Learning Rate: "+ str(self.alpha))
        file.write("\n")
        file.write("L1 constant"+ str(self.L1))
        file.write("\n")
        file.write("L2 constant"+ str(self.L2))
        file.write("\n")
        file.write("Batchnorm: "+ str(self.batchNorm))
        file.write("\n")
        file.write("Dropout prob: "+ str(1-self.dropProb))
        file.write("\n")
        file.write("Training Accuracies: "+ str(self.trackAccuracyTrain))
        file.write("\n")
        file.write("Cross Validation Accuracies: "+ str(self.trackAccuracyCV))
        file.write("\n")
        file.write("Training Costs: "+ str(self.trackCostsTrain))
        file.write("\n")
        file.write("Cross Validation Costs: "+ str(self.trackCostsCV))
        file.write("\n")
        file.write("#########################################")
        file.write("\n")
        file.close()

        saveMatrix(self.weightMatrices, self.name+'weightMatrices')
        saveMatrix(self.biasMatrices, self.name+'biasMatrices')
        return None

    def loadModel(self, weightsFile, biasFile):
        self.weightMatrices = loadMatrix(weightsFile)
        self.biasMatrices = loadMatrix(biasFile)
        return None
