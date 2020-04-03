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
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn
import sklearn.metrics as metrics

#Plot Confusion Matrix for testing and Training Accuracies
def plotConfusionMatrix(confusionMatrix, classes, fname):

	plt.figure(figsize=(10, 7))

	ax = sn.heatmap(confusionMatrix, fmt="d", annot=True, cbar=False,
                    cmap=sn.cubehelix_palette(15),
                    xticklabels=classes, yticklabels=classes)
	# Move X-Axis to top
	ax.xaxis.tick_top()
	ax.xaxis.set_label_position('top')
	
	ax.set(xlabel="Predicted", ylabel="Actual")
	
	
	figure =fname + ".jpg"

	plt.title(fname  , y = 1.08 , loc = "center")
	plt.savefig(figure)
	plt.show()
	plt.close()


def plotROCcurve(testy,proby):
	testy = np.array(testy,dtype='int')/4
	proby = np.array(proby,dtype='float')/4
	print(testy.shape,proby.shape)
	fpr, tpr, threshold = metrics.roc_curve(testy, proby)
	print(fpr,tpr)
	roc_auc = metrics.auc(fpr, tpr)
	plt.title('Receiver Operating Characteristic')
	plt.plot(fpr, tpr, 'b', 'b', label = 'AUC = %0.2f' % roc_auc)
	plt.legend(loc = 'lower right')
	plt.plot([0, 1], [0, 1],'r--')
	plt.xlim([0, 1])
	plt.ylim([0, 1])
	plt.ylabel('True Positive Rate')
	plt.xlabel('False Positive Rate')
	plt.show()



def get_probability(test,dict,dict0,dict4,count,tclass):
    prob = 0.0
    for word in dict:
        if word in test:
            if tclass == 0:
                prob = prob+math.log((dict0[word]+1.)/(dict0[word]+dict4[word]+2.))
            else:
                prob = prob+math.log((dict4[word]+1.)/(dict0[word]+dict4[word]+2.))
        else:
            prob = prob-math.log(dict0[word]+dict4[word]+2.)  
    return prob


def predict(data,dict,dict0,dict4,count):
    y = [] #initialising the prediction array
    j = 0
    prob = []
    for each_x in data:
        j+=1
        print(j)
        predicted_class = 0
        t = 0.0

        max_probab = float("-inf")
        k = np.zeros(2,dtype = 'float')
        min = 0.0
        for i in range(2):
            k[i] = get_probability(each_x,dict,dict0,dict4,count,i)
            if k[i]<min:
            	min = k[i]
        k = k-min
        print(math.exp(k[1])+math.exp(k[0]),k[0],k[1])
        t= math.exp(k[1])/(math.exp(k[1])+math.exp(k[0]))
        if t>0.5:
        	predicted_class = 4
        y.append(predicted_class)
        prob.append(t)
       	print(predicted_class)
    return y,prob

def accuracy(data,dict,dict0,dict4,count,actual):
    prediction,prob = predict(data,dict,dict0,dict4,count)
    print(prob)
    testy = np.array(actual,dtype = 'int')
    predy = np.array(prediction,dtype = 'int')
    proby = np.array(prob,dtype = 'float')
    # plotting confusion matrix
    cm = confusion_matrix(testy, predy, labels=[0,4])
    plotConfusionMatrix(cm,sorted(set(testy)),'confusion_matrix')
    plotROCcurve(testy,proby)
    length = len(prediction)
    sum = 0.0
    for i in range(length):
        if(int(prediction[i]) == testy[i]):
            sum += 1.0
    return ((sum * 100.0) / length)


dict = []
dict0 = []
dict4 = []

# count stores the overall probability
count = np.zeros(2,dtype='uint32')

# reading data from csv file
with open('train.csv', 'r',encoding='latin-1') as csvfile:
    csvreader = csv.reader(csvfile)
    # fields = csvreader.next()
    i = 0
    for row in csvreader:
        i+=1
        if i%10000 == 0:
            print(i)
        tweet = row[5].split()
        if (int(row[0]) == 0):
            count[0]+=1
            # words = set(words)
            for word in tweet:
                dict0.append(word)    
        elif(int(row[0]) == 4):
            count[1]+=1
            # words = set(words)
            for word in tweet:
                dict4.append(word)
        for word in tweet:
            dict.append(word)
        
print(count)


# dictionaries for spam and non-spam
dict = set(dict)
dict0 = Counter(dict0)
dict4 = Counter(dict4)
test = []
actual = []

# testing
with open('test.csv','r',encoding='latin-1') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        if int(row[0]) != 2:
            words = row[5].split()
            test.append(words)
            actual.append(int(row[0]))

acc = accuracy(test,dict,dict0,dict4,count,actual)	
print(acc)