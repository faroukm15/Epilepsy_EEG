from Clustering import K_Means
from NeuralNetwork import *
import numpy as np


class RadialBasis:
    def __init__(self, TrainingData=[], TestingData=[], StoppingCriteria=1, MSEThreshold=-1, Bias=True, eta=0.001,
                 epochs=500, K=3, FeaturesMax = [], FeaturesMin = []):
        self.FeaturesMax = FeaturesMax
        self.FeaturesMin = FeaturesMin
        self.TrainingData = TrainingData
        self.TestingData = TestingData
        self.HiddenSpaceData = []
        self.StoppingCriteria = StoppingCriteria  # 0 For Epochs, 1 For MSE
        self.MSEThreshold = MSEThreshold
        self.K = K
        self.BiasUse = Bias
        self.LearningRate = eta
        self.epochs = epochs
        self.Clusters = 3
        
        if (TrainingData == []):
            return
        
        DataToCluster = []
        for i in range(len(self.TrainingData)):
            DataToCluster.append(self.TrainingData[i][0:len(self.TrainingData) - 2])
        self.ClusteringData = K_Means(DataToCluster, K)
        self.Layers = 3
        NetworkLayers = []
        NeuronsPerLayer = [len(self.TrainingData[0]), K, 1]
        if self.BiasUse == True:
            for i in range(len(NeuronsPerLayer)):
                NeuronsPerLayer[i] = NeuronsPerLayer[i] + 1
        for i in range(self.Layers):
            Neurons = []
            for j in range(NeuronsPerLayer[i]):
                N = Neuron()
                L = []
                if i == self.Layers - 1:
                    NeuronLink = 0
                    L.append(NeuronLink)
                else:
                    if i + 1 == 2:
                        L = np.random.rand(NeuronsPerLayer[i + 1])
                    else:
                        for k in range(NeuronsPerLayer[i + 1]):
                            L.append(0)
                            # for k in range(NeuronsPerLayer[i+1]):
                            # NeuronLink = np.random.rand()
                            # L.append(NeuronLink)
                N.Weights = L
                Neurons.append(N)
            NetworkLayers.append(Layer(Neurons))
        self.Network = Network(NetworkLayers)
        self.ConfusionMatrix = [[0.0 for i in range(2)] for j in range(2)]
        self.OverallAccuracy = 0.0
        for i in self.TrainingData:
            NewData = self.ConvertDataToHiddenSpace(i)
            # NewData.append([len(i)-1])
            self.HiddenSpaceData.append(NewData)
        self.NeuronsPerLayer = NeuronsPerLayer

    def ConvertDataToHiddenSpace(self, Sample):
        NewData = [0.0 for i in range(self.K)]
        ClusterNumber = 0
        for i in self.ClusteringData.Centroids:
            R = self.ClusteringData.EculideanDistance(Sample[0:len(Sample) - 2], i)
            ClusterVariance = self.ClusteringData.CentroidsVariance[ClusterNumber]
            NewData[ClusterNumber] = np.exp((-R ** 2) / (2 * ClusterVariance ** 2))
        return NewData

    def MSECheck(self):
        MSE = 0.0
        for i in range(len(self.HiddenSpaceData)):
            V = 0.0
            for h in range(len(self.Network.Layers[2].Neurons)):
                for x in range(len(self.Network.Layers[1].Neurons)):
                    V += (self.Network.Layers[1].Neurons[x].Value) * (
                        self.Network.Layers[1].Neurons[x].Weights[h])
                self.Network.Layers[2].Neurons[h].Value = V
            Target = self.HiddenSpaceData[i][len(self.HiddenSpaceData[i]) - 1]
            Error = Target - V
            MSE += Error ** 2
        MSE /= 2 * len(self.HiddenSpaceData)
        return MSE

    def TrainModel(self):
        for i in range(self.epochs):
            print('Epoch #', i)
            for j in range(len(self.HiddenSpaceData)):
                for k in range(2, self.Layers):
                    if k == 0:  # Input Layer
                        for h in range(len(self.Network.Layers[k].Neurons)):
                            if h == 0:
                                continue
                            if h == 1 and self.BiasUse:
                                self.Network.Layers[k].Neurons[h].Value = 1
                            self.Network.Layers[k].Neurons[h].Value = self.HiddenSpaceData[i][h]
                        continue
                    V = 0.0
                    for h in range(len(self.Network.Layers[k].Neurons)):
                        for x in range(len(self.Network.Layers[k - 1].Neurons)):
                            V += (self.Network.Layers[k - 1].Neurons[x].Value) * (
                                self.Network.Layers[k - 1].Neurons[x].Weights[h])
                        self.Network.Layers[k].Neurons[h].Value = V
                    Target = self.HiddenSpaceData[j][len(self.HiddenSpaceData[j]) - 1]
                    if V != Target:
                        for f in range(2, self.Layers):
                            for m in range(len(self.Network.Layers[f - 1].Neurons)):
                                for n in range(len(self.Network.Layers[f - 1].Neurons[m].Weights)):
                                    NewWeight = self.Network.Layers[f - 1].Neurons[m].Weights[n] + (
                                        self.LearningRate * self.Network.Layers[f - 1].Neurons[m].Value *
                                        self.HiddenSpaceData[f][n])
                                    self.Network.Layers[f].Neurons[m].Weights[n] = NewWeight
            if self.StoppingCriteria == 1:
                MSE = self.MSECheck()
                if MSE < self.MSEThreshold:
                    break

    def TestModel(self):
        self.ConfusionMatrix = [[0.0 for i in range(2)] for j in range(2)]
        RightPredictions = 0
        for i in range(len(self.TestingData)):
            HiddenSpaceSample = self.ConvertDataToHiddenSpace(self.TestingData[i][0:len(self.TestingData[i]) - 2])
            HiddenSpaceSample.append(self.TestingData[i][-1])
            for j in range(self.Layers):
                if j == 0:
                    continue
                if j == 1:
                    for h in range(len(self.Network.Layers[j].Neurons)):
                        if h == 0 and self.BiasUse:
                            self.Network.Layers[j].Neurons[h].Value = 1
                        self.Network.Layers[j].Neurons[h].Value = HiddenSpaceSample[h]
                    continue
                # for k in range(len(self.Network.Layers[j].Neurons)):
                for h in range(len(self.Network.Layers[j].Neurons)):
                    V = 0.0
                    for x in range(len(self.Network.Layers[j - 1].Neurons)):
                        print ("Val ", self.Network.Layers[j - 1].Neurons[x].Value)
                        print ("ls ", self.Network.Layers[j - 1].Neurons[x].Weights[h])
                        V += (self.Network.Layers[j - 1].Neurons[x].Value) * (
                        self.Network.Layers[j - 1].Neurons[x].Weights[h])
                    self.Network.Layers[j].Neurons[h].Value = V
                Target = HiddenSpaceSample[-1]

                Actual = np.round(self.Network.Layers[self.Layers - 1].Neurons[
                                      0].Value)  # len(self.Network.Layers[self.Layers - 1].Neurons) - 1
                print ("Target ", Target)
                print ("Actual ", Actual)
                if Target == Actual:
                    RightPredictions = RightPredictions + 1
                    self.ConfusionMatrix[int(Target)][int(Target)] = self.ConfusionMatrix[int(Target)][int(Target)] + 1
                else:
                    self.ConfusionMatrix[int(Target)][int(Target) ^ 1] = self.ConfusionMatrix[int(Target)][
                                                                             int(Target) ^ 1] + 1
        self.OverallAccuracy = (float(RightPredictions) / float(len(self.TestingData))) * 100
        print (self.OverallAccuracy)

    def TestNewSample(self, NewSample):
        HiddenSpaceSample = self.ConvertDataToHiddenSpace(NewSample[0:len(NewSample) - 2])
        HiddenSpaceSample.append(NewSample[0:len(self.TestingData) - 1])
        for j in range(self.Layers):
            if j == 0:
                continue
            if j == 1:
                for h in range(len(self.Network.Layers[j].Neurons)):
                    if h == 0 and self.BiasUse:
                        self.Network.Layers[j].Neurons[h].Value = 1
                    self.Network.Layers[j].Neurons[h].Value = self.HiddenSpaceData[h]
                continue
            for k in range(len(self.Network.Layers[j].Neurons)):
                V = 0.0
                for h in range(len(self.Network.Layers[k].Neurons)):
                    for x in range(len(self.Network.Layers[k - 1].Neurons)):
                        V += (self.Network.Layers[k - 1].Neurons[x].Value) * (
                            self.Network.Layers[k - 1].Neurons[x].Weights[h])
                    self.Network.Layers[k].Neurons[h].Value = V
        Actual = self.Network.Layers[self.Layers - 1].Neurons[0].Value
        if Actual == 1:
            return 'Ictal'
        elif Actual == 0:
            return 'Normal'

    def write_model_to_file(self, name='dumm.txt'):
        file = open(name, 'w+')
        file.write('%d\n' % self.K)
        file.write('%d\n' % self.BiasUse)
        file.write('%f\n' % self.LearningRate)
        file.write('%d\n' % self.epochs)
        file.write('%d\n' % self.Clusters)

        for n in self.FeaturesMax:
            file.write('%f ' % n)
        file.write('\n')
        
        for n in self.FeaturesMin:
            file.write('%f ' % n)
        file.write('\n')

        file.write('%d\n' % self.Layers)
        for n in self.NeuronsPerLayer:
            file.write('%d ' % n)
        file.write('\n')
        for i in range(self.Layers):
            for j in range(len(self.Network.Layers[i].Neurons)):
                for k in range(len(self.Network.Layers[i].Neurons[j].Weights)):
                    file.write('%f ' % self.Network.Layers[i].Neurons[j].Weights[k])
                file.write('\n')
        file.write('%f\n' % self.OverallAccuracy)

    def load_model_from_file(self, name):
        file = open(name, 'r')
        self.K = int(file.readline())
        self.BiasUse = int(file.readline())
        self.LearningRate = float(file.readline())
        self.epochs = int(file.readline())
        self.Clusters = int(file.readline())

        #maxF = file.readline()
        #self.FeaturesMax = [float(x) for x in maxF.split()]
        #minF = file.readline()
        #self.FeaturesMin = [float(x) for x in minF.split()]

        self.Layers = int(file.readline())
        tmp = file.readline()
        tmpNeuronsPerLayer = [int(x) for x in tmp.split()]
        self.NeuronsPerLayer = tmpNeuronsPerLayer
        NetworkLayers = []

        for i in range(self.Layers):
            Neurons = []
            for j in range(tmpNeuronsPerLayer[i]):
                tmp = file.readline()
                N = Neuron()
                L = [float(x) for x in tmp.split()]
                N.Weights = L
                Neurons.append(N)
            NetworkLayers.append(Layer(Neurons))
        self.Network = Network(NetworkLayers)
        self.OverallAccuracy = float(file.readline())