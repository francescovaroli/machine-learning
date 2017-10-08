import skimage
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn import svm
from sklearn import datasets, linear_model

x_new = np.load('/home/francesco/ml-project/data/X_train.npy')


import csv
with open('/home/francesco/ml-project/data/y_1.csv', 'r') as f:
    reader = csv.reader(f)
    stringPrediction = list(reader)


def csvList(list):

    counter = len(list)
    predictions = []

    for i in range(0,counter):
        tmp = len(list[i])
        for j in range(0,tmp):
            value = int(list[i][j])
            predictions.append(value)
    return predictions

totalPredictions = csvList(stringPrediction)

def limit(array,paramenter):
    final = []
    for elem in range(0,paramenter):
        final.append(array[elem])
    return final

predictions = limit(totalPredictions,250)

x_shaped= np.reshape(x_new, (-1, 176, 208, 176))

def selectImage(brains,start,end):
    totalBrainImages = []
    xslices = []
    for index in range(start,end):
        for x in (30, 120, 20):
            slice = x_shaped[index][x][:][:]
            xslices.append(slice)
        for y in (40, 180, 20):
            slice = x_shaped[index][:][y][:]
            xslices.append(slice)
        for z in (30, 120, 20):
            slice = x_shaped[index][:][:][z]
            xslices.append(slice)
        totalBrainImages.append(xslices)
    return xslices


    xslices = []
        for y in (40, 180, 20):
            slice = y_shaped[indey][:][y][:]
            yslices.append(slice)

slicesTrain = selectImage(x_shaped,0,250)

slicesTest = selectImage(x_shaped,250,278)


hogImageTraining = []

for n in range(0,250):

    slice = slicesTrain[n]

    fd, hog_image = hog(slice, orientations=9, pixels_per_cell=(4, 4),
                        cells_per_block=(3, 3), feature_vector=True,visualise=True)
    hogImageTraining.append(fd)







hogImageTesting = []
for i in range(0,len(slicesTest)):

    sliceTest = slicesTest[i]
    fd, hog_image = hog(sliceTest, orientations=9, pixels_per_cell=(4, 4),
                        cells_per_block=(3, 3), visualise=True)
    hogImageTesting.append(fd)


regr= linear_model.LinearRegression()
regr.fit(hogImageTraining,predictions)
prediction= regr.predict(hogImageTesting)

clf = svm.SVC()
clf.fit(hogImageTraining,predictions)

prediction = clf.predict(hogImageTesting)

hTrain = [i *1000 for i in hogImageTraining]
minimum = min(min(hTrain[0]),min(hTrain[0]))
maximum = max(max(hTrain[53]),max(hTrain[53]))

bins = np.linspace(minimum,maximum,300)
hist = plt.hist(hTrain[0],bins,alpha = 0.5,label = "young")
hist = plt.hist(hTrain[53],bins,alpha = 0.5,label = "old")

for i in range(28):
    print(hogImageTesting[i][100000])

plt.show()