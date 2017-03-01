import pyspark
import sqlite3
import numpy as np
import string
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import sys
import io
import re
import json


from pyspark import SparkContext
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy
from textblob import TextBlob
from subprocess import Popen, PIPE

if __name__ == "__main__":
    sc = SparkContext(appName="TweetSentiment")
    def format_sentence(sentence):
        return {word:True for word in word_tokenize(sentence)}
    
    
    #Sample data for negative and positive sentiments
    pos_data=[]
    cat = Popen(["hadoop", "fs", "-cat", "/user/surbhi/Twitter/pos.txt"],stdout=PIPE)
    for line in cat.stdout:
        pos_data.append([format_sentence(line.decode('latin-1')),'pos'])

    neg_data=[]
    cat1 = Popen(["hadoop", "fs", "-cat", "/user/surbhi/Twitter/neg.txt"],stdout=PIPE)
    for line in cat1.stdout:
        neg_data.append([format_sentence(line.decode('latin-1')),'neg'])

    #Divide the data into 2 sets training and testing data.
    training_data=pos_data[:4000]+ neg_data[:4000]
    testing_data=pos_data[4000:]+ neg_data[4000:]
    
    #train your model using training data
    model= NaiveBayesClassifier.train(training_data)


    neg=0
    pos=0
    nneg=0
    npos=0
    total=0
    neutral=0
    file = Popen(["hadoop", "fs", "-cat", "/user/surbhi/Twitter/raw_tweets.json"],stdout=PIPE)
    str=  file.stdout.read()
    str.encode('latin-1')
    d=json.loads(str)
    for content in d:
        #Data preperation : data cleaning and removal of stopwords
        sentence=content['text']
        sentence = sentence.lower()
        sentence = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',sentence) #Convert www.* or https?://* to URL
        sentence = re.sub(r'<[^>]+>',"",sentence) # HTML tags
        sentence = re.sub(r'(?:@[\w_]+)',"",sentence) # @-mentions
        sentence = re.sub(r'(?:(?:\d+,?)+(?:\.?\d+)?)', "",sentence) #numbers
        sentence = re.sub('(\s+)(a|an|and|the)(\s+)', " ", sentence)
        sentence = re.sub(r'#([^\s]+)', r'\1', sentence)#Replace #word with word
        sentence = sentence.strip('\'"?,.')#strip punctuation
        sentence = sentence.strip('\'"')#trim

        #Method 1 - By using Naive Bayes Classifier
        sentiment=model.classify(format_sentence(sentence))
        if sentiment=='pos':
            npos=npos+1
        else:
            nneg=nneg+1
            
        #Method 2 - By using in-built function in python textblob package
        analysis=TextBlob(sentence);
        if analysis.sentiment[0]<=0 :
            neg=neg+1
        else:
            pos=pos+1
        total=total+1
        
    #Calculating percentage of negative and positive reviews
    neg_per=float(neg*100)/total
    pos_per=float(pos*100)/total
    nneg_per=float(nneg*100)/total
    npos_per=float(npos*100)/total
    
    print "Using TextBlob : negative = %s(%0.2f%%) positive = %s(%0.2f%%) "%(neg,neg_per,pos,pos_per)
    print "Using Naive Bayes Algorithm : negative = %s(%0.2f%%) positive = %s(%0.2f%%) "%(nneg,nneg_per,npos,npos_per)
    performance = [neg,pos]
    
    # data to plot
    n_groups = 2
    txt_blob = (neg,pos)
    nb = (nneg,npos)

    # create plot
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.35
    opacity = 0.8

    rects1 = plt.bar(index, txt_blob, bar_width,
                     alpha=opacity,
                     color='b',
                     label='TextBlob')

    rects2 = plt.bar(index + bar_width, nb, bar_width,alpha=opacity,
                     color='g',
                     label='Naive Bayes')

    plt.xlabel('Sentiments')
    plt.ylabel('Number of Tweets')
    plt.title('SSentiment Analysis')
    plt.xticks(index + bar_width, ('Negative', 'Positive'))
    plt.legend()

    plt.tight_layout()
    plt.savefig('output.png')
    os.system('hadoop fs -copyFromLocal output.png  /user/path/to/hdfs_folder/output.png') #store plot in hdfs 
    plt.show()
    sc.stop()



