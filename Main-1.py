from DataSetManager import DataSet
import numpy as np
import random
from BackPropagation import BackPropagation
from RadialBasisFunction import RadialBasis


def TestNewSample(FilePath):
    NewSample = []
    with open(FilePath) as f:
        lines = f.readlines()
        for Line in lines:
            NewSample = list(map(float, Line.strip().split()))
    return NewSample


def test_radial(D):
    starter_eta = 0.001
    epochs_1 = 500
    epochs_2 = 1000
    for i in range(20):
        R = RadialBasis(D.TrainingData, D.TestingData, 0, 1, False, starter_eta, epochs_1, 3)
        R.TrainModel()
        file_name = 'Result\\R' + str(i) + '_NoMSE_NoB' + '.txt'
        R.TestModel()
        R.write_model_to_file(file_name)

        R = RadialBasis(D.TrainingData, D.TestingData, 1, 1, False, starter_eta, epochs_1, 3)
        R.TrainModel()
        file_name = 'Result\\R' + str(i) + '_MSE_NoB' + '.txt'
        R.TestModel()
        R.write_model_to_file(file_name)

        R = RadialBasis(D.TrainingData, D.TestingData, 0, 1, True, starter_eta, epochs_1, 3)
        R.TrainModel()
        file_name = 'Result\\R' + str(i) + '_NoMSE_B' + '.txt'
        R.TestModel()
        R.write_model_to_file(file_name)

        R = RadialBasis(D.TrainingData, D.TestingData, 1, 1, True, starter_eta, epochs_1, 3)
        R.TrainModel()
        file_name = 'Result\\R' + str(i) + '_MSE_B' + '.txt'
        R.TestModel()
        R.write_model_to_file(file_name)
        starter_eta += 0.001

    starter_eta = 0.001
    for i in range(2, 10):
        R = RadialBasis(D.TrainingData, D.TestingData, 0, 1, True, starter_eta, epochs_1, i)
        R.TrainModel()
        file_name = 'Result2\\R' + str(i) + '_NoMSE_B' + '.txt'
        R.TestModel()
        R.write_model_to_file(file_name)

        R = RadialBasis(D.TrainingData, D.TestingData, 1, 1, True, starter_eta, epochs_1, 3)
        R.TrainModel()
        file_name = 'Result2\\R' + str(i) + '_MSE_B' + '.txt'
        R.TestModel()
        R.write_model_to_file(file_name)

        starter_eta += 0.001

def test_BP(D):
    starter_eta = 0.001
    epochs_1 = 500
    epochs_2 = 1000
    randomNeurons = [i for i in range(30)]
    k = 0
    for i in range(1, 21):
        # E , M, C for stopping criteria
        # L1, L2. L3, L4, L5 for number of layers
        # s for sigmoid
        NeuronsPerLayer = []
        for j in range(i):
            NeuronsPerLayer.append(random.choice(randomNeurons))
        #print NeuronsPerLayer

        B = BackPropagation(D.TrainingData, D.ValidationData, D.TestingData, 0, 0, i, NeuronsPerLayer, -1, False, 0.001, 500)
        B.TrainModel()
        file_name = 'Result3\\B' + str(k) + 'L' + str(i) + '_S_E_NoB' + '.txt'
        B.TestModel()
        B.write_model_to_file(file_name)

        k += 1

        B = BackPropagation(D.TrainingData, D.ValidationData, D.TestingData, 0, 0, i, NeuronsPerLayer, -1, False, 0.001,
                            1000)
        B.TrainModel()
        file_name = 'Result3\\B' + str(k) + 'L' + str(i) + '_S_E_NoB' + '.txt'
        B.TestModel()
        B.write_model_to_file(file_name)

        k += 1

        B = BackPropagation(D.TrainingData, D.ValidationData, D.TestingData, 0, 1, i, NeuronsPerLayer, -1, False, 0.001,
                            500)
        B.TrainModel()
        file_name = 'Result3\\B' + str(k) + 'L' + str(i) + '_S_M_NoB' + '.txt'
        B.TestModel()
        B.write_model_to_file(file_name)

        k += 1

        B = BackPropagation(D.TrainingData, D.ValidationData, D.TestingData, 0, 2, i, NeuronsPerLayer, -1, False, 0.001,
                            500)
        B.TrainModel()
        file_name = 'Result3\\B' + str(k) + 'L' + str(i) + '_S_C_NoB' + '.txt'
        B.TestModel()
        B.write_model_to_file(file_name)

        k += 1

        B = BackPropagation(D.TrainingData, D.ValidationData, D.TestingData, 0, 0, i, NeuronsPerLayer, -1, True, 0.001,
                            1000)
        B.TrainModel()
        file_name = 'Result3\\B' + str(k) + 'L' + str(i) + '_S_E_B' + '.txt'
        B.TestModel()
        B.write_model_to_file(file_name)

        k += 1

        B = BackPropagation(D.TrainingData, D.ValidationData, D.TestingData, 0, 1, i, NeuronsPerLayer, -1, True, 0.001,
                            1000)
        B.TrainModel()
        file_name = 'Result3\\B' + str(k) + 'L' + str(i) + '_S_M_B' + '.txt'
        B.TestModel()
        B.write_model_to_file(file_name)

        k += 1

        B = BackPropagation(D.TrainingData, D.ValidationData, D.TestingData, 0, 1, i, NeuronsPerLayer, -1, True, 0.001,
                            1000)
        B.TrainModel()
        file_name = 'Result3\\B' + str(k) + 'L' + str(i) + '_S_C_B' + '.txt'
        B.TestModel()
        B.write_model_to_file(file_name)

        k += 1

    #-----------------------------------------------------------------------------------------------------------

        B = BackPropagation(D.TrainingData, D.ValidationData, D.TestingData, 1, 0, i, NeuronsPerLayer, -1, False, 0.001,
                            500)
        B.TrainModel()
        file_name = 'Result3\\B' + str(k) + 'L' + str(i) + '_T_E_NoB' + '.txt'
        B.TestModel()
        B.write_model_to_file(file_name)

        k += 1

        B = BackPropagation(D.TrainingData, D.ValidationData, D.TestingData, 1, 0, i, NeuronsPerLayer, -1, False, 0.001,
                            1000)
        B.TrainModel()
        file_name = 'Result3\\B' + str(k) + 'L' + str(i) + '_T_E_NoB' + '.txt'
        B.TestModel()
        B.write_model_to_file(file_name)

        k += 1

        B = BackPropagation(D.TrainingData, D.ValidationData, D.TestingData, 1, 1, i, NeuronsPerLayer, -1, False, 0.001,
                            500)
        B.TrainModel()
        file_name = 'Result3\\B' + str(k) + 'L' + str(i) + '_T_M_NoB' + '.txt'
        B.TestModel()
        B.write_model_to_file(file_name)

        k += 1

        B = BackPropagation(D.TrainingData, D.ValidationData, D.TestingData, 1, 2, i, NeuronsPerLayer, -1, False, 0.001,
                            500)
        B.TrainModel()
        file_name = 'Result3\\B' + str(k) + 'L' + str(i) + '_T_C_NoB' + '.txt'
        B.TestModel()
        B.write_model_to_file(file_name)

        k += 1

        B = BackPropagation(D.TrainingData, D.ValidationData, D.TestingData, 1, 0, i, NeuronsPerLayer, -1, True, 0.001,
                            1000)
        B.TrainModel()
        file_name = 'Result3\\B' + str(k) + 'L' + str(i) + '_T_E_B' + '.txt'
        B.TestModel()
        B.write_model_to_file(file_name)

        k += 1

        B = BackPropagation(D.TrainingData, D.ValidationData, D.TestingData, 1, 1, i, NeuronsPerLayer, -1, True, 0.001,
                            1000)
        B.TrainModel()
        file_name = 'Result3\\B' + str(k) + 'L' + str(i) + '_T_M_B' + '.txt'
        B.TestModel()
        B.write_model_to_file(file_name)

        k += 1

        B = BackPropagation(D.TrainingData, D.ValidationData, D.TestingData, 1, 1, i, NeuronsPerLayer, -1, True, 0.001,
                            1000)
        B.TrainModel()
        file_name = 'Result3\\B' + str(k) + 'L' + str(i) + '_T_C_B' + '.txt'
        B.TestModel()
        B.write_model_to_file(file_name)

        k += 1

def main():
    #D = DataSet('F:\\FCIS\\Year 4\\Semester 1\\Computational Intelligence\\Labs\\Slides\\Lab12 - Project Support\\Dataset')
    D = DataSet('F:\\kolia\\year4\\first_term\\computional intelligence\\project\\NNProject-V2\\Dataset')
    #B = BackPropagation(D.TrainingData, D.ValidationData, D.TestingData, 0, 0, 5, [2, 3, 4, 5, 6])
    #B.ConstructInputLayer(D.TrainingData)

    #B.TrainModel()
    #B.write_model_to_file('out.txt')
    #R = RadialBasis(D.TrainingData, D.TestingData)
    #R.TrainModel()
    #R.write_model_to_file('out.txt')
    #R.TestModel()
    #test_radial(D)
    #R.load_model_from_file('R0_NoMSE_B.txt')
    #R.TestModel()
    test_BP(D)


#if __name__ == "__main__":
main()
