import numpy as np
import math
import csv
import time
from cvxopt import matrix
from cvxopt import solvers

start = time.time()
def gaussian(x,z,gamma):
    f = x-z
    fmag = np.dot(f,f)
    fmag = -fmag*gamma
    return(math.exp(fmag))


def train(trainxi,trainxj,trainyi,trainyj):
    datax = np.append(trainxi,trainxj,axis = 0)
    datay = np.append(trainyi,trainyj)
    row,col = datax.shape
    # setting parameters to call solver function 
    G = np.zeros(2*row*row).reshape(2*row,row)
    h = np.zeros(2*row)
    for i in range (row):
        G[i][i] = -1.
        G[i+row][i] = 1.
        h[i+row] = 1.
    G = matrix(G)
    h = matrix(h)
    A = matrix(datay,(1,row),tc = 'd')
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

    # calculating b only from obtained parameters
    maxneg = 0.0
    minpos = 0.0
    for i in range (row):
        temp = 0.0
        if int(datay[i]) == -1:
            for j in range (row):
                temp += alpha[j]*datay[j]*gaussian(datax[i],datax[j],gamma)
            if  temp>maxneg:
                maxneg = temp
        elif int(datay[i]) == 1:
            for j in range (row):
                temp += alpha[j]*datay[j]*gaussian(datax[i],datax[j],gamma)
            if temp<minpos:
                minpos = temp

    b = -(maxneg+minpos)/2.
    return alpha,b

def predict(trainxi,trainxj,trainyi,trainyj,alpha,b,sv,x):
    datax = np.append(trainxi,trainxj)
    datay = np.append(trainyi,trainyj)
    pred = b
    gamma = 0.05
    for j in sv:
        pred+= alpha[j]*datay[j]*gaussian(datax[j],x,gamma)
    return pred

# data extraction from csv file

trainx = []
datax = []
alphas = []
bs = []
svs = []
for i in range (10):
    trainx.append(np.array([[]]))
    datax.append([])
with open('data/train.csv','r') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        datax[int(float(row[784]))].append(row[0:784])
for i in range (10):
    trainx[i] = np.array(datax[i],dtype = float)/255.

for i in range(1,10):
    for j in range(i):
        trainyi = np.ones(len(trainx[i]),dtype= 'float')
        trainyj = -np.ones(len(trainx[j]),dtype= 'float')
        alpha,b = train(trainx[i],trainx[j],trainyi,trainyj)
        
        # support vectors
        sv = []
        for i in range(len(alpha)):
            if alpha > 0.000001:
                sv.append(i)
        svs.append(sv)
        alphas.append(alpha)
        bs.append(b)

print('training completed!!!')

testx = []
testy = []
tdatax = []
with open('data/test.csv','r') as csvfile:
    csvreader = csv.reader(csvfile)
    for trow in csvreader:
        testx.append(trow[0:784])
        testy.append(float(trow[784]))
testx = np.array(testx,dtype = 'float')/255.
testy = np.array(testy,dtype = 'float')

tlen,twid = testx.shape
print(testx.shape)
tpred = 0
for t in range(tlen):
    n = 0
    count = np.zeros(10)
    score = np.zeros(10,dtype='float')
    for i in range(1,10):
        for j in range(i):
            trainyi = np.ones(len(trainx[i]),dtype= 'float')
            trainyj = -np.ones(len(trainx[j]),dtype= 'float')
            val = predict(trainx[i],trainx[j],trainyi,trainyj,alphas[n],bs[n],svs[n],testx[t])
            n+=1
            if (val >= 0.):
                count[i]+=1
                score[i]+=val
            elif (val<=-0.):
                count[j]+=1
                score[j]+=(-val)
    pred = 0
    temp = 0
    for i in range(10):
        if (count[i]>temp):
            pred = i
        elif(count[i]==temp):
            if (score[i]>score(pred)):
                pred = i
    if pred == testy[t]:
        tpred +=1

print(tpred*100./tlen)

end = time.time()
print(end-start)