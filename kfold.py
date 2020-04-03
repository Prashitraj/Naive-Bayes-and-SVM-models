import numpy as np
import math
import csv
import time
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold


def train(traind,testd):
    trainx = (traind[:,0:784])/255.
    trainy = traind[:,784]
    testx = (testd[:,0:784])/255.
    testy = testd[:,784]
    svm = SVC(C=0.001,gamma = 0.05,decision_function_shape='ovo').fit(trainx,trainy)
    print('training completed!!!')
    accuracy = svm.score(testx,testy)
    print(accuracy)
    return accuracy

start = time.time()
# data extraction from csv file
data = []
with open('data/train.csv','r') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        data.append(row)
data = np.array(data,dtype = 'float')    
kf = KFold(n_splits=5, shuffle = True)
kf.get_n_splits(data)
avg = 0
for train_index, test_index in kf.split(data):
    train_data = data[train_index]
    test_data = data[test_index]
    accuracy = train(train_data,test_data)
    avg += accuracy

print(avg/5.)
end = time.time()
print(end-start)