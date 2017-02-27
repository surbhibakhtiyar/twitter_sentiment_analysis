import pyspark
import sqlite3
import numpy as np
import string
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import sys
import io
import re
import json
import plotly.plotly as py
import plotly.graph_objs as go


from pyspark import SparkContext
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.classify import NaiveBayesClassifier, MaxentClassifier, SklearnClassifier
from nltk.classify.util import accuracy
from textblob import TextBlob
from sklearn import svm
from subprocess import Popen, PIPE

if __name__ == "__main__":
    sc = SparkContext(appName="TweetSentiment")
    def format_sentence(sentence):
        return {word:True for word in word_tokenize(sentence)}

    pos_data=[]
    with io.open('rt-polarity-pos.txt',encoding='latin-1') as f:
        for line in f:
            pos_data.append([format_sentence(line),'pos'])

    neg_data=[]
    with io.open('rt-polarity-neg.txt',encoding='latin-1') as f:
        for line in f:
            neg_data.append([format_sentence(line),'neg'])


    training_data=pos_data[:4000]+ neg_data[:4000]
    testing_data=pos_data[4000:]+ neg_data[4000:]

    model= NaiveBayesClassifier.train(training_data)


    neg=0
    pos=0
    nneg=0
    npos=0
    total=0
    neutral=0
    file = open('raw_tweets.json').read()
    #pprint(file)
    info = json.loads(file)
    for content in info:
        sentence=content['text']
        #sentence="Congrats @ravikiranj, i heard you wrote a new tech post on sentiment analysis"
        #Convert to lower case
        sentence = sentence.lower()
        #Convert www.* or https?://* to URL
        sentence = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',sentence)
        sentence = re.sub(r'<[^>]+>',"",sentence) # HTML tags
        sentence = re.sub(r'(?:@[\w_]+)',"",sentence) # @-mentions
        sentence = re.sub(r'(?:(?:\d+,?)+(?:\.?\d+)?)', "",sentence) #numbers
        sentence = re.sub('(\s+)(a|an|and|the)(\s+)', " ", sentence)
        #Replace #word with word
        sentence = re.sub(r'#([^\s]+)', r'\1', sentence)
        #strip punctuation
        sentence = sentence.strip('\'"?,.')
        #trim
        sentence = sentence.strip('\'"')


        #print sentence


        sentiment=model.classify(format_sentence(sentence))
        if sentiment=='pos':
            npos=npos+1
        else:
            nneg=nneg+1

        analysis=TextBlob(sentence);
        if analysis.sentiment[0]<=0 :
            neg=neg+1
        else:
            pos=pos+1
        total=total+1
    print "negative = %s positive = %s "%(neg,pos)

    print "negative = %s positive = %s "%(nneg,npos)
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
    plt.show()
    sc.stop()



