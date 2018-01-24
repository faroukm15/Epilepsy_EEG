import DataSetManager as datasetManager
from BackPropagation import BackPropagation
from RadialBasisFunction import *
from tkinter import *

# Form Decleration
root = Tk()
root.title('NN Project')
root.geometry('580x540+0+0')

# Training Data
global dataset
global backProb
global rbf

backProb = BackPropagation()
rbf = RadialBasis()


# Form Parameters
datasetLocation = StringVar()
activationFunction = IntVar()
stoppingCritria = IntVar()
modelType = IntVar()
layersCount = IntVar()
neuronsCount = StringVar()
clustersCount = IntVar()
mseThreshold = DoubleVar()
bias = IntVar()
eta = DoubleVar()
epochs = IntVar()
sampleTest = StringVar()
modelPath = StringVar()

# Form Design
# 1- Dataset location
lblDatasetLocation = Label(root, text="Dataset Location", font=('arial', 12), fg='steelblue')
lblDatasetLocation.place(x=11, y=10)
txtDatasetLocation = Entry(root, textvariable=datasetLocation, width=65, bg='lightgreen')
txtDatasetLocation.place(x=170, y=14)

# 2- activation function
lblActivationFunction = Label(root, text="Activation Function: ", font=('arial', 12), fg='steelblue')
lblActivationFunction.place(x=11, y=45)
radSigm = Radiobutton(root, text="Sigmoid", font=('arial', 12), fg='steelblue', variable=activationFunction, value=0)
radSigm.place(x=155, y=43)
radTanh = Radiobutton(root, text="TanH", font=('arial', 12), fg='steelblue', variable=activationFunction, value=1)
radTanh.place(x=245, y=43)

# 3- stopping critria
def stoppingCritria_onChange():
    if (stoppingCritria.get() == 1):
        txtMSEThreshold.config(state='normal')
    else:
        txtMSEThreshold.config(state='disabled')
        mseThreshold = -1

lblStoppingCritria = Label(root, text="Stopping Critria: ", font=('arial', 12), fg='steelblue')
lblStoppingCritria.place(x=11, y=85)
radEpochs = Radiobutton(root, text="Epochs", font=('arial', 12), fg='steelblue', variable=stoppingCritria, command = stoppingCritria_onChange, value=0)
radEpochs.place(x=155, y=83)
radMSE = Radiobutton(root, text="MSE", font=('arial', 12), fg='steelblue', variable=stoppingCritria, command = stoppingCritria_onChange, value=1)
radMSE.place(x=245, y=83)
radCV = Radiobutton(root, text="Cross Val.", font=('arial', 12), fg='steelblue', variable=stoppingCritria, command = stoppingCritria_onChange, value=2)
radCV.place(x=315, y=83)

#4- BP or RB
def modelType_onChange():
    if (modelType.get()):
        txtLayersCount.config(state='disabled')
        txtNeuronsCount.config(state='disabled')
        txtClustersCount.config(state='normal')
    else:
        txtLayersCount.config(state='normal')
        txtNeuronsCount.config(state='normal')
        txtClustersCount.config(state='disabled')

lblModelType = Label(root, text="Model Type: ", font=('arial', 12), fg='steelblue')
lblModelType.place(x=11, y=125)
radBP= Radiobutton(root, text="Backpropagation", font=('arial', 12), fg='steelblue', variable=modelType, command = modelType_onChange, value=0)
radBP.place(x=155, y=123)
radRBF = Radiobutton(root, text="RBF", font=('arial', 12), fg='steelblue', variable=modelType, command = modelType_onChange, value=1)
radRBF.place(x=315, y=123)


# 5- number of layers
lblLayersCount = Label(root, text="Layers Count: ", font=('arial', 12), fg='steelblue')
lblLayersCount.place(x=11, y=160)
txtLayersCount = Entry(root, textvariable=layersCount, width=10, bg='lightgreen')
txtLayersCount.place(x=120, y=163)
# 6 neurons per layer
lblNeuronsCount = Label(root, text="Neurons Count: ", font=('arial', 12), fg='steelblue')
lblNeuronsCount.place(x=315, y=160)
txtNeuronsCount = Entry(root, textvariable=neuronsCount, width=20, bg='lightgreen')
txtNeuronsCount.place(x=435, y=163)

#7- clusturs
lblClustersCount = Label(root, text="Clusters Count: ", font=('arial', 12), fg='steelblue')
lblClustersCount.place(x=11, y=205)
txtClustersCount = Entry(root, textvariable=clustersCount, width=10, bg='lightgreen')
txtClustersCount.place(x=120, y=208)
txtClustersCount.config(state='disabled')

# 8- MSE Threashold
lblMSEThreshold = Label(root, text="MSE Threshold: ", font=('arial', 12), fg='steelblue')
lblMSEThreshold.place(x=11, y=245)
txtMSEThreshold = Entry(root, textvariable=mseThreshold, width=10, bg='lightgreen')
txtMSEThreshold.place(x=120, y=248)
txtMSEThreshold.config(state='disabled')

#9- bias
lblBias = Label(root, text="Bias Value: ", font=('arial', 12), fg='steelblue')
lblBias.place(x=11, y=275)
txtBias = Entry(root, textvariable=bias, width=10, bg='lightgreen')
txtBias.place(x=120, y=278)

# 10- learning rate
lblEta = Label(root, text="Learning Rate: ", font=('arial', 12), fg='steelblue')
lblEta.place(x=195, y=275)
txtEta = Entry(root, textvariable=eta, width=10, bg='lightgreen')
txtEta.place(x=315, y=278)

# 11- numbers of epochs
lblEpochs = Label(root, text="Epochs: ", font=('arial', 12), fg='steelblue')
lblEpochs.place(x=415, y=275)
txtEpochs = Entry(root, textvariable=epochs, width=10, bg='lightgreen')
txtEpochs.place(x=485, y=278)

# 12 - Buttons
def trainModel():
    global dataset
    global backProb
    global rbf
    
    dataset = datasetManager.DataSet(datasetLocation.get())
    if (modelType.get() == 0):
        _neuronsCount = [int(x) for x in neuronsCount.get().split(',')]
        backProb = BackPropagation(dataset.TrainingData,dataset.ValidationData,dataset.TestingData,activationFunction.get(),stoppingCritria.get(),layersCount.get(), _neuronsCount,mseThreshold.get(), bias.get(), eta.get(),epochs.get())
        backProb.FeaturesMax = dataset.FeatureMax
        backProb.FeaturesMin = dataset.FeatureMin
        backProb.TrainModel()
        backProb.write_model_to_file()
    else:
        rbf = RadialBasis(dataset.TrainingData, dataset.TestingData, stoppingCritria.get(), mseThreshold.get(), bias.get(), eta.get(), epochs.get(), clustersCount.get())
        rbf.FeaturesMax = dataset.FeatureMax
        rbf.FeaturesMin = dataset.FeatureMin
        rbf.TrainModel()
        rbf.write_model_to_file()

def testModel():
    global backProb
    global rbf

    if (modelType.get() == 0):
        backProb.TestModel()
        print(backProb.OverallAccuracy)
        print(backProb.ConfusionMatrix)
    else:
        rbf.TestModel()
        print(rbf.OverallAccuracy)
        print(rbf.ConfusionMatrix)
        


btnTrain = Button(root, text="Train Model", bg='lightblue', command=trainModel).place(x=215,y=305)
btnTest = Button(root, text="Test Model", bg='lightblue', command=testModel).place(x=305,y=305)

# 13 - Load Model
def LoadModel():
    global backProb
    global rbf
    if (modelType.get() == 0):
        backProb.load_model_from_file(modelPath.get())
    else:
        rbf.load_model_from_file(modelPath.get())

lblModel = Label(root, text="Model", font=('arial', 12), fg='steelblue')
lblModel.place(x=11, y=405)
txtModelPath = Entry(root, textvariable=modelPath, width=75, bg='lightgreen').place(x=110, y=408)
btnLoadModel = Button(root, text="Load Model", bg='lightblue', command=LoadModel).place(x=485,y=435)

# 14- sample test
def TestNewSample():
    global dataset
    global backProb
    global rbf

    NewSample = []
    with open(sampleTest.get()) as f:
        lines = f.readlines()
        for Line in lines:
            NewSample.append(list(map(float, Line.strip().split())))
    
    if(modelType.get() == 0):
        normalData = datasetManager.Min_Max_normalization_param(NewSample, backProb.FeaturesMax, backProb.FeaturesMin)
        for v in normalData:
            result = backProb.TestNewSample(v)
            print(result)
    else:
        normalData = NewSample #datasetManager.Min_Max_normalization_param(NewSample, rbf.FeaturesMax, rbf.FeaturesMin)
        for v in normalData:
            result = rbf.TestNewSample(v)
            print(result)
    #return NewSample



lblSampleTest = Label(root, text="SampleTest", font=('arial', 12), fg='steelblue')
lblSampleTest.place(x=11, y=465)
txtSampleTest = Entry(root, textvariable=sampleTest, width=75, bg='lightgreen').place(x=110, y=468)
btnTestSample = Button(root, text="Test Sample", bg='lightblue', command=TestNewSample).place(x=485,y=495)

root.mainloop()
