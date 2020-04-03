import numpy as np
import csv
import time 
from sklearn.svm import SVC
from cvxopt import matrix
from cvxopt import solvers

start = time.time()
# data extraction from csv file
datax = []
datay = []
with open('data/train.csv','r') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        if int(float(row[784])) == 9:
            datax.append(row[0:784])
            datay.append(1.)
        elif int(float(row[784])) == 0:
            datax.append(row[0:784])
            datay.append(-1.)
            
datax = np.array(datax,dtype='float')
datax = datax/255.
datay = np.array(datay,dtype='float')


testx = []
testy = []
with open('data/test.csv','r') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        if int(float(row[784])) == 9:
            testx.append(row[0:784])
            testy.append(1.)
        elif int(float(row[784])) == 0:
            testx.append(row[0:784])
            testy.append(-1.)
            
testx = np.array(testx,dtype='float')
testx = testx/255.
testy = np.array(testy)
testy.astype('float')

# setting parameters to call solver function 
svm = SVC(kernel = 'rbf',C= 1,gamma =0.05).fit(datax, datay)

accuracy = svm.score(testx,testy)
print(accuracy)
pred = svm.predict(testx)
count = 0
for i in range(len(testy)):
    if pred[i] == testy[i]:
        count+=1
print(count)

# calculatiing total time taken to run the program
end = time.time()
print(end-start)