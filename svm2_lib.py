import numpy as np
import math
import csv
import time
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn

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

start = time.time()
# data extraction from csv file
datax = []
datay = []
with open('data/train.csv','r') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        datax.append(row[0:784])
        datay.append(float(row[784]))    
datax = np.array(datax,dtype='float')/255.
datay = np.array(datay,dtype = 'float')
row,col = datax.shape
print(datax.shape)
print(datay.shape)

svm = SVC(C=1,gamma = 0.05,decision_function_shape='ovo').fit(datax,datay)

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

dec = svm.decision_function(testx)
accuracy = svm.score(testx,testy)
predy = svm.predict(testx)
predy.astype('int')
testy.astype('int')
print(accuracy)
cm = confusion_matrix(testy, predy, labels=[0,1,2,3,4,5,6,7,8,9])
plotConfusionMatrix(cm,sorted(set(testy)),'confusion_matrix')

end = time.time()
print(end-start)