import math
import sys


class K_Means:
    def __init__(self,DataSet, K = 3, Tolerance = 0.0001, MaxIterations = 500):
        self.K = K
        self.Tolerance = Tolerance
        self.MaxIterations = MaxIterations
        self.ClustersData = [[] for i in range(self.K)]
        self.Centroids = [0 for i in range(self.K)]
        self.Data = DataSet
        self.CentroidsVariance = [0.0 for i in range(self.K)]
        for i in range(self.K):
            self.Centroids[i] = self.Data[i]
        self.ConstructClusters()

    def EculideanDistance(self, Centroid, Feature):
        SquaredDistance = 0.0
        for i in range(len(Centroid)):
            SquaredDistance += (Centroid[i] - Feature[i])**2
        return math.sqrt(SquaredDistance)

    def ComputeCentroids(self):
        for i in range(len(self.Centroids)):
            Mean = [0.0 for j in range(self.Centroids[i])]
            for j in range(len(self.ClustersData[i])):
                for k in range(len(self.ClustersData[i][0])):
                    Mean[k] += self.ClustersData[i][j][k]
            Mean = [x/len(self.ClustersData[i]) for x in Mean]
            self.Centroids[i] = Mean

    def ComputeCentroidsVariance(self):
        for i in range(self.K):
            for j in self.ClustersData[i]:
                SampleVar = 0.0
                for k in range(len(j)):
                    SampleVar += abs(self.ClustersData[i][j][k]-self.Centroids[i][k])
                self.CentroidsVariance += SampleVar
            self.CentroidsVariance /= len(self.ClustersData[i])

    def ConstructClusters(self):
        for i in range(self.MaxIterations):
            self.ClustersData = [[] for i in range(self.K)]
            for j in range(len(self.Data)):
                MinDistance = sys.maxint
                Cluster = -1
                for k in range(len(self.Centroids)):
                    Tmp = self.EculideanDistance(self.Data[j],self.Centroids[k])
                    if Tmp < MinDistance:
                        MinDistance = Tmp
                        Cluster = k
                self.ClustersData[Cluster].append(self.Data[j])
            OldCentroids = [[0 for j in range(len(self.Centroids[0]))] for i in range(len(self.Centroids))]
            for i in range(len(self.Centroids)):
                for j in range(len(self.Centroids[i])):
                    OldCentroids[i][j] = self.Centroids[i][j]
            self.ComputeCentroids()
            Continue = False
            for k in range(len(self.Centroids)):
                for y in range(len(self.Centroids[i])):
                    if abs(OldCentroids[k][y] - self.Centroids[k][y]) > self.Tolerance:
                        Continue = True
                        break
            if not Continue:
                break
        self.ComputeCentroidsVariance()