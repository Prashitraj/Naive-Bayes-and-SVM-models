import numpy as np
import csv
from sklearn.svm import SVC
import time

start = time.time()
# data extraction from csv file
datax = []
datay = []
with open('data/train.csv','r') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        if int(float(row[784])) == 9:
            datax.append(row[0:784])
            datay.append(1)
        elif int(float(row[784])) == 0:
            datax.append(row[0:784])
            datay.append(-1)
            
datax = np.array(datax,dtype='float')
datax = datax/255.
datay = np.array(datay)
datay.astype('float')
row,col = datax.shape
print(datax.shape)
print(datay.shape)

# setting parameters to call solver function 
svm = SVC(kernel='linear',C= 1)
svm.fit(datax, datay)

testx = []
testy = []
with open('data/test.csv','r') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        if int(float(row[784])) == 9:
            testx.append(row[0:784])
            testy.append(1)
        elif int(float(row[784])) == 0:
            testx.append(row[0:784])
            testy.append(-1)
            
testx = np.array(testx,dtype='float')
testx = testx/255.
testy = np.array(testy)
testy.astype('float')

accuracy = svm.score(testx,testy)
print(accuracy)

end = time.time()
print(end-start)