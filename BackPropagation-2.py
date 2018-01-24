from NeuralNetwork import *
import numpy as np
from CrossValidation import CrossValidation
import random

class BackPropagation:
    def __init__(self, TrainingData=[], ValidationData=[], TestingData=[], ActivationFunction=0, StoppingCriteria=0, Layers=1, NeuronsPerLayer=[1], MSEThreshold = -1, Bias = False, eta=0.001, epochs=500):
        self.LearningRate = eta
        self.epochs = epochs
        self.Sigma = []
        self.BiasUse = Bias
        self.TrainingData = TrainingData
        self.ValidationData = ValidationData
        self.TestingData = TestingData
        if StoppingCriteria != 2:
            for i in range(len(self.ValidationData)):
                self.TrainingData.append(self.ValidationData[i])
            
        self.MSEThrshold = MSEThreshold
        self.CrossValidationValues = CrossValidation()
        self.ActivationFunction = ActivationFunction        #0 For Sigmoid, 1 For Tanh
        self.StoppingCriteria = StoppingCriteria            #0 For Epochs, 1 For MSE, 2 For Cross Validation
        self.Layers = Layers + 2
        NetworkLayers = []
        NeuronsPerLayer.insert(0, 20 + (1 if self.BiasUse else 0))
        NeuronsPerLayer.append(1) # number of neurons in output layer
        if self.BiasUse == True:
            for i in range(len(NeuronsPerLayer)):
                NeuronsPerLayer[i] = NeuronsPerLayer[i]+1
        for i in range(self.Layers):
            Neurons = []
            for j in range(NeuronsPerLayer[i]):
                N = Neuron()
                L = []
                if i == self.Layers-1:
                    NeuronLink = 0
                    L.append(NeuronLink)
                else:
                    L = np.random.rand(NeuronsPerLayer[i + 1])
                    #for k in range(NeuronsPerLayer[i+1]):
                        #NeuronLink = np.random.rand()
                        #L.append(NeuronLink)
                N.Weights = L
                Neurons.append(N)
            NetworkLayers.append(Layer(Neurons))
        self.Network = Network(NetworkLayers)
        self.ConfusionMatrix = [[0.0 for i in range(2)] for j in range(2)]
        self.OverallAccuracy = 0.0

    def Sigmoid(self, X):
        return 1.0/(1.0+np.exp(-X))

    def Tanh(self, X):
        return ((np.exp(X) - np.exp(-X))/(np.exp(X) + np.exp(-X)))

    def Activate(self, V):
        if self.ActivationFunction == 0:
            return self.Sigmoid(V)
        elif self.ActivationFunction == 1:
            return self.Tanh(V)

    def ComputeInputSignal(self, Sample):
        for k in range(self.Layers):
            if k == 0:  # Input Layer
                
                for h in range(len(self.Network.Layers[k].Neurons)):
                    if h == 0 and self.BiasUse:
                        self.Network.Layers[k].Neurons[h].Value = 1
                        self.Network.Layers[k].Neurons[h].input = 1
                        continue
                    self.Network.Layers[k].Neurons[h].Value = Sample[h- (1 if self.BiasUse else 0) ]
                    self.Network.Layers[k].Neurons[h].input = Sample[h- (1 if self.BiasUse else 0) ]
                continue
            for h in range(len(self.Network.Layers[k].Neurons)):
                V = 0.0
                for x in range(len(self.Network.Layers[k - 1].Neurons) ):
                    V += (self.Network.Layers[k - 1].Neurons[x].Value) * (self.Network.Layers[k - 1].Neurons[x].Weights[h])
                Y = self.Activate(V)
                self.Network.Layers[k].Neurons[h].Value = Y
                self.Network.Layers[k].Neurons[h].input = V

    def GetActivationDerivative(self, Actual):
        if self.ActivationFunction == 0:
            sig = self.Sigmoid(Actual)
            return sig*(1.0-sig)
        elif self.ActivationFunction == 1:
            return (np.exp(-Actual))/(1+(np.exp(-Actual))*(np.exp(-Actual)))

    def ComputeErrorSignal(self, Target):
        self.Sigma = [[] for i in range(self.Layers)]
        for i in range(self.Layers-1, 0, -1):
            LayerSigma = []
            for j in range(len(self.Network.Layers[i].Neurons)):
                if self.BiasUse and j == 0:
                    LayerSigma.append(0)
                    continue
                if i == self.Layers-1:
                    Actual = self.Network.Layers[i].Neurons[j].Value
                    inp = self.Network.Layers[i].Neurons[j].input
                    Error = Target - Actual
                    LayerSigma.append(Error*self.GetActivationDerivative(inp))
                else:
                    Actual = self.Network.Layers[i].Neurons[j].Value
                    inp = self.Network.Layers[i].Neurons[j].input
                    NeuronSigma = self.GetActivationDerivative(inp)
                    Error = 0.0
                    for k in range(len(self.Network.Layers[i+1].Neurons)):
                       Error += (self.Sigma[i+1][k])*(self.Network.Layers[i].Neurons[j].Weights[k])
                    LayerSigma.append(NeuronSigma*Error)
            self.Sigma[i]=LayerSigma

    def UpdateWeights(self):
        for i in range(1, self.Layers):
            for j in range(len(self.Network.Layers[i-1].Neurons)):
                for k in range(len(self.Network.Layers[i].Neurons)):
                    NewWeight = self.Network.Layers[i-1].Neurons[j].Weights[k] + (self.LearningRate * self.Network.Layers[i-1].Neurons[j].Value * self.Sigma[i][k])
                    self.Network.Layers[i-1].Neurons[j].Weights[k] = NewWeight

    def MSECheck(self, Data):
        MSE = 0.0
        for i in range(len(Data)):
            for k in range(1, self.Layers):
                V = 0.0
                for h in range(len(self.Network.Layers[k].Neurons)):
                    for x in range(len(self.Network.Layers[k - 1].Neurons)):
                        V += (self.Network.Layers[k - 1].Neurons[x].Value) * (self.Network.Layers[k - 1].Neurons[x].Weights[h])
                    Y = self.Activate(V)
                    self.Network.Layers[k].Neurons[h].Value = Y
            MSE += (self.Network.Layers[self.Layers-1].Neurons[0].Value - Data[i][len(Data[i])-1])**2
        MSE /= 2*len(Data)
        return MSE

    def TrainModel(self):
        for i in range(self.epochs):
            random.shuffle(self.TrainingData)
            for j in range(len(self.TrainingData)):
                self.ComputeInputSignal(self.TrainingData[j])
                self.ComputeErrorSignal(self.TrainingData[j][len(self.TrainingData[j])-1])
                self.UpdateWeights()
                #Check Stopping
                if self.StoppingCriteria == 1:
                    MSEValue = self.MSECheck(self.TrainingData)
                    if MSEValue < self.MSEThrshold:
                        break
                elif self.StoppingCriteria == 2 and i+1 % 50 == 0:
                    MSEValue = self.MSECheck(self.ValidationData)
                    if MSEValue > self.CrossValidationValues.LeastMSE:
                        break
                    else:
                        self.CrossValidationValues.LeastMSE = MSEValue
                        for h in range(self.Layers):
                            NetworkLinks =[]
                            for x in range(len(self.Network.Layers[h].Neurons)):
                                NetworkLinks.append(self.Network.Layers[h].Neurons[x].Weights)
                            self.CrossValidationValues.Weights.append(NetworkLinks)

    def TestModel(self):
        RightPredictions = 0
        for i in range(len(self.TestingData)):
            for j in range(self.Layers):
                if j == 0:
                    for h in range(len(self.Network.Layers[j].Neurons)):
                        if h == 0 and self.BiasUse:
                            self.Network.Layers[j].Neurons[h].Value = 1
                            continue
                        self.Network.Layers[j].Neurons[h].Value = self.TestingData[i][h- (1 if self.BiasUse else 0) ]
                    continue
                for h in range(len(self.Network.Layers[j].Neurons)):
                    V = 0.0
                    for x in range(len(self.Network.Layers[j - 1].Neurons)):
                        V += (self.Network.Layers[j - 1].Neurons[x].Value) * (self.Network.Layers[j - 1].Neurons[x].Weights[h])
                    Y = self.Activate(V)
                    self.Network.Layers[j].Neurons[h].Value = Y
            
            Target = self.TestingData[i][len(self.TestingData[i])-1]
            Actual = self.Network.Layers[self.Layers-1].Neurons[0].Value
            if Actual >= 0.5 :
                Actual = 1 
            else :
                Actual = 0
                
            if Target == Actual:
                RightPredictions = RightPredictions+1
                self.ConfusionMatrix[Target][Target] = self.ConfusionMatrix[Target][Target]+1
            else:
                self.ConfusionMatrix[Target][Target^1] = self.ConfusionMatrix[Target][Target^1]+1
        print("Right " + str(RightPredictions))
        print("All "+str(len(self.TestingData)))
        self.OverallAccuracy = (RightPredictions / float(len(self.TestingData)))*100

    def TestNewSample(self, NewSample):
        for j in range(self.Layers):
            if j == 0:
                for h in range(len(self.Network.Layers[j].Neurons)):
                    if h == 0 and self.BiasUse:
                        self.Network.Layers[j].Neurons[h].Value = 1
                        continue
                    self.Network.Layers[j].Neurons[h].Value = NewSample[h- (1 if self.BiasUse else 0) ]
                continue
            for k in range(len(self.Network.Layers[j].Neurons)):
                V = 0.0
                for h in range(len(self.Network.Layers[j].Neurons)):
                    for x in range(len(self.Network.Layers[j - 1].Neurons)):
                        V += (self.Network.Layers[j - 1].Neurons[x].Value) * (
                        self.Network.Layers[j - 1].Neurons[x].Weights[h])
                    Y = self.Activate(V)
                    self.Network.Layers[j].Neurons[h].Value = Y

        Actual = self.Network.Layers[self.Layers-1].Neurons[0].Value
        if Actual >= 0.5:
            Actual = 1
        else:
            Actual = 0

        if Actual == 1:
            return 'Ictal'
        elif Actual == 0:
            return 'Normal'
