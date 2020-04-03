import numpy as np
import csv
import re
import sys
import pickle
import math
from collections import Counter
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.naive_bayes import GaussianNB


tweets = []
trainy = []
# count stores the overall probability
count = np.zeros(2,dtype='uint32')
with open('train.csv', 'r',encoding = 'latin-1') as csvfile:
    csvreader = csv.reader(csvfile)
    # fields = csvreader.next()
    i = 0
    for row in csvreader:
        i+=1
        if i%10000 == 0:
            print(i)
        tweets.append(row[5])
        trainy.append(row[0])    

trainy = np.array(trainy, dtype = 'int')/4

vectorizer = TfidfVectorizer(dtype= np.float32, min_df = 1000)
print("completed")
trainx = vectorizer.fit_transform(tweets)
print(trainx.shape)
selector = SelectKBest(chi2,k = 300)
selector.fit(trainx, trainy)
trainx = selector.transform(trainx)
print(trainx.shape)
#print(X.shape)
testx = []
testy = []
print("training")
# testing
with open('test.csv','r',encoding = 'latin-1') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        if int(row[0]) != 2:
            testx.append(row[5])
            testy.append(int(row[0]))
testy = np.array(testy,dtype = 'int')/4
clf = GaussianNB()
clf.fit(trainx.toarray(), trainy)
testx = vectorizer.transform(testx)
testx = selector.transform(testx)
acc = clf.score(testx.toarray(), testy)
print(acc)