class FeatureExtractor:
    def __init__(self, SampleData, Class):
        self.IQR = 0.0
        self.MAD = 0.0
        self.Covariance = 0.0
        self.Class = Class
        self.Size = len(SampleData)
        self.Energy = 0.0
        self.Mean = 0.0
        self.Std = 0.0
        self.SampleData = SampleData
        self.ExtractFeatures()

    def GetSampleMean(self):
        for i in self.SampleData:
            self.Mean += i
        self.Mean /= self.Size

    def ExtractFeatures(self):
        self.GetSampleMean()
        SortedData = self.SampleData
        for i in range(self.Size):
            self.Energy += (self.SampleData[i] ** 2)
            self.MAD += (self.SampleData[i] - self.Mean)
            self.Std += ((self.SampleData[i] - self.Mean) ** 2)
        self.MAD /= self.Size
        self.Std /= self.Size
        self.Covariance = (self.Std / (self.Mean ** 2))
        Median = SortedData[(self.Size / 2) + 1]
        MedinaIndex = SortedData.index(Median)
        Q1Data = SortedData[0:MedinaIndex - 1]
        Q2Data = SortedData[MedinaIndex + 1:self.Size]
        Q1 = 0.0
        Q2 = 0.0
        if len(Q1Data) % 2 == 0:
            Q1 = Q1Data[len(Q1Data) / 2]
        else:
            Q1 = (Q1Data[len(Q1Data) / 2] + Q1Data[(len(Q1Data) / 2) + 1]) / 2
        if len(Q2Data) % 2 == 0:
            Q2 = Q2Data[len(Q2Data) / 2]
        else:
            Q2 = (Q2Data[len(Q2Data) / 2] + Q2Data[(len(Q2Data) / 2) + 1]) / 2
        self.IQR = Q2 - Q1
        self.WriteExtractedFeatures()

    def WriteExtractedFeatures(self):
        print(str(self.Energy) + "\t\t\t\t\t" + str(self.MAD) + "\t\t\t\t\t" + str(self.Covariance) + "\t\t\t\t\t" + str(self.IQR) + "\t\t\t\t\t" + self.Class)