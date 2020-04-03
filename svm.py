import numpy as np
import csv
from cvxopt import matrix
from cvxopt import solvers
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
G = np.zeros(2*row*row).reshape(2*row,row)
h = np.zeros(2*row)
for i in range (row):
    G[i][i] = -1.
    G[i+row][i] = 1.
    h[i+row] = 1.
G = matrix(G)
h = matrix(h)
A = matrix(datay,(1,4500),tc = 'd')
b = matrix(0.0)
P = np.zeros(row*row).reshape(row,row)
for i in range (row):
    for j in range (row):
        P[i][j] = datay[i]*datay[j]*np.dot(datax[i],datax[j])
P= matrix(P)
q = -np.ones(row)
q = matrix(q)
sol = solvers.qp(P,q,G,h,A,b)
alpha = sol['x']

sv = []
threshold = 0.000001
for i in range(row):
    if(alpha[i]>threshold):
        sv.append(i)
w = np.zeros(784,dtype = 'float')

# calculating w and b from obtained parameters
for i in sv:
    w += alpha[i]*datay[i]*datax[i]
maxneg = -float("inf")
minpos = float("inf")
for i in range (row):
    temp = 0
    if int(datay[i]) == -1:
        temp = (np.dot(w,datax[i]))
        if  temp>maxneg:
            maxneg = temp
    elif int(datay[i]) == 1:
        temp = (np.dot(w,datax[i]))
        if temp<minpos:
            minpos = temp

b = -(maxneg+minpos)/2.
print(b)
print(maxneg,minpos)
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

trows,tcols = testx.shape
count = 0
for i in range(trows):
    pred = np.dot(w,testx[i])+b
    if pred*testy[i] >= 0.:
        count+=1
print(count*100./trows)

end = time.time()
print(end-start)