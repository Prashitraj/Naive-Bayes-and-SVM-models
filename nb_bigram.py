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

# stemming function
def get_stemmed_tweet(tweet):
    # initialising stemmer
    tokenizer = RegexpTokenizer(r'\W+')
    en_stop = set(stopwords.words('english'))
    p_stemmer = PorterStemmer()
    out = []
    tweet = tweet.lower()
    words = re.split(r'\W+', tweet)
    stopped_tokens = [token for token in words if (token not in en_stop and not token.startswith('@') and len(token)>1 )]
    out = [p_stemmer.stem(token) for token in stopped_tokens]
    return out

def bigram_feature_vector(word_list, k):
    if(k == 1):
        return word_list
    new_list = []
    for i in range(len(word_list) - k + 1):
        some_string = ''
        for j in range(k):
            some_string += word_list[i + j] + ' '
        new_list.append(some_string.strip())
    return new_list

def get_probability(test,dict,dict0,dict4,count,tclass):
    prob = math.log((count[tclass]+1.)/(np.sum(count)+2.))
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
    for each_x in data:
        j+=1
        print(j)
        predicted_class = '0'
        max_probab = float("-inf")
        for i in range(2):
            curr_prob = get_probability(each_x,dict,dict0,dict4,count,i)
            if(curr_prob > max_probab):
                predicted_class = i
                max_probab = curr_prob
        y.append(predicted_class)
    return y

def accuracy(data,dict,dict0,dict4,count,actual):
    prediction = predict(data,dict,dict0,dict4,count)
    length = len(prediction)
    sum = 0.0
    for i in range(length):
        if(int(prediction[i]*4) == actual[i]):
            sum += 1.0
    return ((sum * 100.0) / length)


dict = []
dict0 = []
dict4 = []

# count stores the overall probability
count = np.zeros(2,dtype='uint32')

# reading data from csv file
with open('train.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    # fields = csvreader.next()
    i = 0
    for row in csvreader:
        i+=1
        if i%10000 == 0:
            print(i)
        tweet = get_stemmed_tweet(row[5])
        # bigram feature vector
        tweet = kgram_feature_vector(tweet,2)
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
with open('test.csv','r') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        if int(row[0]) != 2:
            words = get_stemmed_tweet(row[5])
            #bigram feature vector 
            words = bigram_feature_vector(words,2)
            test.append(words)
            actual.append(int(row[0]))

acc = accuracy(test,dict,dict0,dict4,count,actual)
print(acc)
