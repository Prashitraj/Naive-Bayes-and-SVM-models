import numpy as np
import math
import csv
import time
from cvxopt import matrix
from cvxopt import solvers

def gaussian(x,z,gamma):
    f = np.subtract(x,z)
    fmag = np.dot(f,f)
    fmag = -fmag*gamma
    return(math.exp(fmag))

start = time.time()
# data extraction from csv file
datax = []
datay = []
with open('data/train.csv','r') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        if int(float(row[784])) == 1:
            datax.append(row[0:784])
            datay.append(1)
        elif int(float(row[784])) == 2:
            datax.append(row[0:784])
            datay.append(-1)
            
datax = np.array(datax,dtype='float')/255.
datay = np.array(datay,dtype= 'float')
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
gamma = 0.05
for i in range (row):
    for j in range (row):
        P[i][j] = datay[i]*datay[j]*gaussian(datax[i],datax[j],gamma)
P= matrix(P)
q = -1.*np.ones(row)
q = matrix(q)
sol = solvers.qp(P,q,G,h,A,b)
alpha = sol['x']
sv = []
threshold = 0.000001
for i in range(row):
    if(alpha[i]>threshold):
        sv.append(i)
print(len(sv))
# calculating b only from obtained parameters
maxneg = -float("inf")
minpos = float("inf")
for i in range (row):
    if int(datay[i]) == -1:
        temp = 0.0
        for j in sv:
            temp += alpha[j]*datay[j]*gaussian(datax[i],datax[j],gamma)
        if  temp>maxneg:
            maxneg = temp
    elif int(datay[i]) == 1:
        temp = 0.0
        for j in sv:
            temp += alpha[j]*datay[j]*gaussian(datax[i],datax[j],gamma)
        if temp<minpos:
            minpos = temp

b = -(maxneg+minpos)/2.

print(b)
print(maxneg,minpos)

testx = []

testy = []
with open('data/test.csv','r') as csvfile:
    csvreader = csv.reader(csvfile)
    for trow in csvreader:
        if int(float(trow[784])) == 1:
            testx.append(trow[0:784])
            testy.append(1)
        elif int(float(trow[784])) == 2:
            testx.append(trow[0:784])
            testy.append(-1)
            
testx = np.array(testx,dtype='float')
testx = testx/255.
testy = np.array(testy)
testy.astype('float')
trows,tcols = testx.shape
count = 0
for i in range(trows):
    pred= b
    for j in sv:
        pred+= alpha[j]*datay[j]*gaussian(datax[j],testx[i],gamma)
    if pred*testy[i] > 0.:
        count+=1
print(count*100./trows)
end = time.time()
print(end-start)