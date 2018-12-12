import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O
from sklearn.model_selection import train_test_split # function for splitting data to train and test set

import nltk # Sentiment anaylsis package
from nltk.corpus import stopwords
from nltk.classify import SklearnClassifier

from wordcloud import WordCloud,STOPWORDS
import matplotlib.pyplot as plt
%matplotlib inline

from subprocess import check_output

# Splitting the dataset into train and test set
train, test = train_test_split(data,test_size = 0.1)

# Removing neutral sentiments
train = train[train.sentiment != "Neutral"]

train_pos = train[ train['sentiment'] == 'Positive']
train_pos = train_pos['text']
train_neg = train[ train['sentiment'] == 'Negative']
train_neg = train_neg['text']

def wordcloud_draw(data, color = 'black'):
    words = ' '.join(data)
    cleaned_word = " ".join([word for word in words.split()
                            if 'http' not in word
                                and not word.startswith('@')
                                and not word.startswith('#')
                                and word != 'RT'
                            ])
    wordcloud = WordCloud(stopwords=STOPWORDS,
                      background_color=color,
                      width=2400,
                      height=1900
                     ).generate(cleaned_word)
    plt.figure(1,figsize=(13, 13))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()
    
print("******Here is the Positive wordcloud after applying sentiment analysis approach*******")
wordcloud_draw(train_pos,'white')
print("*****Here is the Negative wordcloud after applying sentiment analysis approach*******")
wordcloud_draw(train_neg)
