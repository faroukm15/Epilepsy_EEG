from FeatureExtractor import FeatureExtractor
import sys
import os
import numpy as np
import random 

def Min_Max_normalization(X):
    feature_min = [];
    feature_max = [];

    for i in range(20):
        current_min = X[0][i]
        for x in X:
            if x[i] < current_min:
                current_min = x[i]
        current_max = X[0][i]
        for x in X:
            if x[i] >  current_max:
                current_max = x[i]
        feature_min.append(current_min)
        feature_max.append(current_max)
        for j in range(len(X)):
            X[j][i] = (X[j][i]-current_min)/(current_max-current_min)
    return X, feature_max, feature_min

def Min_Max_normalization_param(X, feature_Max, feature_Min):
    for i in range(20):
        current_min = feature_Min[i]
        current_max = feature_Max[i]
        for j in range(len(X)):
            X[j][i] = (X[j][i]-current_min)/(current_max-current_min)
    return X

class DataSet:
    def __init__(self, DataSetPath):
        self.DataSetPath = DataSetPath
        self.FilesChar = 'FNOSZ'
        self.DataSetFeatures = []
        self.TrainingData = []
        self.TestingData = []
        self.ValidationData = []
        self.FeatureMin = []
        self.FeatureMax = []
        self.DataDivision()

    def DataDivision(self):
        Class = -1
        for FileType in self.FilesChar:
            if FileType == 'Z' or FileType == 'O':
                Class = 0
            else:
                Class = 1
            CurrentPath = self.DataSetPath + '\\' + FileType + '.txt'
            lines = None
            with open(CurrentPath) as f:
                all_data = [] 
                lines = f.readlines()
                i = 0
                for Line in lines:
                    Sample = list(map(float, Line.strip().split()))
                    Sample.append(Class)
                    all_data.append(Sample)
                random.shuffle(all_data)
                for Sample in all_data:
                    if i < 240:
                        self.TrainingData.append(Sample)
                    elif i < 300:
                        self.ValidationData.append(Sample)
                    elif i < 400:
                        self.TestingData.append(Sample)
                    i = i+1
                
        self.TrainingData, feature_max, feature_min = Min_Max_normalization(self.TrainingData)
        self.FeatureMin = feature_min
        self.FeatureMax =feature_max
        self.ValidationData = Min_Max_normalization_param(self.ValidationData,feature_max,feature_min)
        self.TestingData= Min_Max_normalization_param(self.TestingData,feature_max,feature_min)