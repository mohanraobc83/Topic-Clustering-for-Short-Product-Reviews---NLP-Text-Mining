# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 23:04:44 2017

@author: Mohan Rao B C & Naveen Pyneni
"""
#####################################################################################################################
#################Trying to develop a Topic clustering to identify the topics at sentence ############################
#####################################################################################################################

# Topic Modeling - Product Reviews - IIMC - APDS
## Importing Required Libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
from nltk.stem import WordNetLemmatizer
from sklearn.cluster import KMeans
import re
import csv
import nltk.data
import os
import num2words
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import TextBlob

## importing Raw data and converting reviews to sentences

# Loading Tokenizer to tokenize reviews to sentences

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

#Setting Working Directory

os.chdir('C:/Users/lenovo/Desktop/APDS/Project/Work/')

#Initiating DataFrame

df = pd.DataFrame()
with open('Amazon_Reviews_ConsolidatedV1.0.csv', newline='') as f:
    reader = csv.reader(f)
    i = 1
    key = 1
    for row in reader:
        if i == 1:
            i = i + 1
            continue
        row_6 = re.sub('\?\?\?\?\?',"",row[6])
        row_6 = re.sub('\-\-+', "", row_6)
        row_6 = re.sub('\?', "", row_6)
        row_6 = re.sub('\.\ \.', ".", row_6)
        row_6 = re.sub('\#', "", row_6)
        row_6 = re.sub('\.\,', ". ", row_6)
        row_6 = re.sub('\*', "", row_6)
        row_6 = re.sub('\;\)', "happy", row_6)
        row_6 = re.sub('\;\-\)', "happy", row_6)
        row_6 = re.sub('\:\-\)', "happy", row_6)
        row_6 = re.sub('\:\D', "happy", row_6)
        row_6 = re.sub('\:\(', "sad", row_6)
        row_6 = re.sub('\.\)', "", row_6)
        row_6 = re.sub('\.\.\.\.+', ". ", row_6)
        row_6 = re.sub('\.\.+', ". ", row_6)
        row_6 = re.sub('\.\ \.+', ". ", row_6)
        row_6 = re.sub('\.+', ". ", row_6)
        row_6 = re.sub('\?\?+',"",row_6)
        row_6 = re.sub('\!+',"",row_6)
        row_6 = re.sub('\.\.\?\?\?\?',"",row_6)
        row_6 = re.sub('\:\)',"",row_6)
        row_6 = re.sub('\,\.',"",row_6)
        row_6 = re.sub('\.\?', ". ", row_6)
        row_6 = re.sub('\[\.\ \ \]', "", row_6)
        row_6 = re.sub(r"(\d+)", lambda x: num2words.num2words(int(x.group(0))), row_6)
        row_6 = tokenizer.tokenize(row_6)
        for sent in row_6:
            if i == 2:
                df = pd.DataFrame({'Date': row[1], 'URL': row[2], 'Review_Title': row[3], 'Author': row[4], 'Rating': row[5], 'Review_text': row[6], 'Review_helpful': row[7], 'Key': key, 'Sentence': sent}, index=[0])
                i = i + 1
                continue
            df2 = pd.DataFrame({'Date': row[1], 'URL': row[2], 'Review_Title': row[3], 'Author': row[4], 'Rating': row[5], 'Review_text': row[6], 'Review_helpful': row[7], 'Key': key, 'Sentence': sent}, index=[0])
            df = df.append(df2, ignore_index=True)
        key = key + 1
df = df[['Date',	'Key',	'URL',	'Review_helpful',	'Rating', 'Author','Review_Title', 'Review_text', 'Sentence']]
df.to_csv("ConvertedFile_amazon1.csv", index=False)


##Import the data

# Import the excel file and call it xls_file
df = pd.read_csv('ConvertedFile_amazon1.csv', encoding = "ISO-8859-1")



#converting the reviews into a list

docs = df['Sentence'].tolist()

# we need to ignore the shorter reviews with less than 3 words as they dont add value to the task of finding themes
# initiating an empty list to get the length of the review in terms of number of words
frequency = []

# for loop to get the list of review lengths in terms of number of words
for docs in docs:
    x=docs.split(None)
    f= len(x)
    frequency.append(f)

# adding the review lengths in terms of number of words to the initial dataframe
    
se=pd.Series(frequency)
df['review_length'] = se.values

# Filtering out the reviews that have less than 4 words

df1 = df[df['review_length'] > 0]

docs1 = df1['Sentence'].tolist()

#applying stemmer

#docs3 = [" ".join([stem(word) for word in sentence.split(" ")]) for sentence in docs1]

#applying stemmer
#wordnet_lemmatizer = WordNetLemmatizer()
#docs3 = [" ".join([wordnet_lemmatizer.lemmatize(word) for word in sentence.split(" ")]) for sentence in docs]

# Function for cleaning documents

no_features = 10000


# NMF is able to use tf-idf
tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=10, max_features=no_features, stop_words='english', ngram_range=(1,2))
tfidf = tfidf_vectorizer.fit_transform(docs1)
tfidf_feature_names = tfidf_vectorizer.get_feature_names()


# LDA can only use raw term counts for LDA because it is a probabilistic graphical model
tf_vectorizer = CountVectorizer(max_df=0.90, min_df=10, max_features=no_features, stop_words='english', ngram_range=(1,2))
tf = tf_vectorizer.fit_transform(docs1)
tf_feature_names = tf_vectorizer.get_feature_names()



no_topics = 50

# Run NMF

nmf = NMF(n_components=no_topics, random_state=1, alpha=.1, l1_ratio=.6, init='nndsvd').fit(tfidf)

# Run LDA

lda = LatentDirichletAllocation(n_topics=no_topics, max_iter=500, learning_method='online', learning_offset=10.,random_state=0).fit(tf)

# Run LSA

lsa = TruncatedSVD(n_components = no_topics, n_iter=500).fit(tfidf)

# Run Kmeans

kmeans = KMeans(n_clusters=no_topics, init='k-means++', max_iter=500, n_init=1).fit(tfidf)
labels = kmeans.labels_

#function to print the topics

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()

#Printing topic numbers and the top words

no_top_words = 10
print_top_words(nmf, tfidf_feature_names, no_top_words)
print_top_words(lsa, tfidf_feature_names, no_top_words)
print_top_words(lda, tf_feature_names, no_top_words)


#Extracting Topic numbers - LDA
doc_topic_lda = lda.transform(tf)
LDA_topic_n = []
LDA_topic_pr = []
for n in range(doc_topic_lda.shape[0]):
    topic_most_pr = doc_topic_lda[n].argmax()
    LDA_topic_n.append(n)
    LDA_topic_pr.append(topic_most_pr)

#Extracting Topic numbers - LSA
doc_topic_lsa = lsa.transform(tfidf)

LSA_topic_n = []
LSA_topic_pr = []
for n in range(doc_topic_lsa.shape[0]):
    topic_most_pr = doc_topic_lsa[n].argmax()
    print("doc: {} topic: {}\n".format(n,topic_most_pr))
    LSA_topic_n.append(n)
    LSA_topic_pr.append(topic_most_pr)

#Extracting Topic numbers - NMF
doc_topic_nmf = nmf.transform(tfidf)
NMF_topic_n = []
NMF_topic_pr = []
for n in range(doc_topic_nmf.shape[0]):
    topic_most_pr = doc_topic_nmf[n].argmax()
    print("doc: {} topic: {}\n".format(n,topic_most_pr))
    NMF_topic_n.append(n)
    NMF_topic_pr.append(topic_most_pr)

#adding all the topic numbers to the initial dataframe for exploratory analysis

m = pd.Series(LDA_topic_pr)

n = pd.Series(LSA_topic_pr)

o = pd.Series(NMF_topic_pr)

df["LDA"] = m.values
  
df["LSA"] = n.values

df["NMF"] = o.values

df["KMeans"] = labels

#Exporting the resulting file in to a CSV file
  
df.to_csv('Amazon_50TopicsV2.csv', sep=',')



###### Performing Sentiment Analysis on the review sentences ###########

docs = df['Sentence'].tolist()

custom_Stopword = ['i','me','my','myself','we','our','ours','ourselves','you','your','yours','yourself','yourselves','he','him','his','himself','she','her','hers','herself','it','its','itself','they','them','their','theirs','themselves','what','which','who','whom','this','that','these','those','am','is','are','was','were','be','been','being','have','has','had','having','do','does','did','doing','a','an','the','and','but','if','or','because','as','until','while','of','at','by','for','with','about','against','between','into','through','during','before','after','above','below','to','from','up','down','in','out','on','off','over','under','again','further','then','once','here','there','when','where','why','how','all','any','both','each','few','more','most','other','some','such','only','own','same','so','than','too','very','s','can','will','just','should','now']

sentiment = []

for docs in docs:
    sentence1_Words = word_tokenize(docs)
    filtered_words = [word for word in sentence1_Words if word not in custom_Stopword]
    sentence1 = ' '.join(filtered_words)
    senti_sentence1 = TextBlob(sentence1)
    ys = senti_sentence1.sentiment.polarity
    sentiment.append(ys)
df["Sentiment"] = sentiment

df.to_csv('Amazon_50Topics_SentimentV1.0.csv', sep=',')
