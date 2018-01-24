from DataSetManager import DataSet
from BackPropagation import BackPropagation


def TestNewSample(FilePath):
    NewSample = []
    with open(FilePath) as f:
        lines = f.readlines()
        for Line in lines:
            NewSample = list(map(float, Line.strip().split()))
    return NewSample


def main():
    D = DataSet('C:\\Users\\Farouk\Desktop\\NNProject-V2\\Dataset\\')
    B = BackPropagation(D.TrainingData,D.ValidationData,D.TestingData,0,0,1, [2], 0, False, 0.1,1)
    #print(B.TrainingData)
    # B.ConstructInputLayer(D.TrainingData)
    B.TrainModel()
    B.TestModel()
    print(B.OverallAccuracy)
    print(B.ConfusionMatrix)
if __name__ == "__main__":
    main()
