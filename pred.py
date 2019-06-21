# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 12:31:04 2019

@author: chitt
"""

import pandas as pd
import numpy as np
from nltk.corpus import wordnet
import string
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer
from os import path
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import pickle
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from utils import train_pipeline_utils


#df = pd.read_csv('resources//data//test_generic.csv')
#df["is_bad_review"] = df["classified"].apply(lambda x: -1 if x =='negative' else 0 if x=='neutral' else 1)
#
#print(df.shape)
#print(df.groupby('is_bad_review').count())
#reviews_df = pd.DataFrame()
#reviews_df[['is_bad_review','review']] = df[['is_bad_review','review']]
#reviews_df = reviews_df.astype(str)
## clean text data
#print(reviews_df["review"].head(5))
#preprocess = train_pipeline_utils.Train_Pipeline_Util()
#reviews_df["review_clean"] = reviews_df["review"].apply(lambda x: preprocess.clean_text(x))
#
#print(reviews_df.shape)
#print(reviews_df.head()) 
#reviews_df.to_csv("resources//data//test_generic_clean1.csv")
#print('Clean file saved')

#reviews_df = pd.read_csv("resources//data//test_generic_clean1.csv")
#print(reviews_df.shape)
#sid = SentimentIntensityAnalyzer()
#reviews_df["sentiments"] = reviews_df["review"].apply(lambda x: sid.polarity_scores(x))
#reviews_df = pd.concat([reviews_df.drop(['sentiments'], axis=1), reviews_df['sentiments'].apply(pd.Series)], axis=1)
#
#print(reviews_df.shape)
#
## add number of characters column
#reviews_df["nb_chars"] = reviews_df["review"].apply(lambda x: len(x))
## add number of words column
#reviews_df["nb_words"] = reviews_df["review"].apply(lambda x: len(x.split(" ")))
#
#print(reviews_df.shape)
#reviews_df["review_clean"] = reviews_df["review_clean"].astype(str)
#reviews_df.to_csv('resources//data//test_generic_SentimentIntensityAnalyzer.csv')

#reviews_df = pd.read_csv('resources//data//test_generic_SentimentIntensityAnalyzer.csv')
#print(reviews_df.shape)
#
#with open('resources//pickle//Doc2Vec_model.pkl','rb') as rp:
#    model = pickle.load(rp)
#
#doc2vec_df = reviews_df["review_clean"].apply(lambda x: model.infer_vector(x.split(" "))).apply(pd.Series)
#doc2vec_df.columns = ["doc2vec_vector_" + str(x) for x in doc2vec_df.columns]
#reviews_df = pd.concat([reviews_df, doc2vec_df], axis=1)
#print(reviews_df.shape)
#reviews_df.to_csv('resources//data//test_generic_doc2vec.csv')


reviews_df = pd.read_csv('resources//data//test_generic_doc2vec.csv')
print(reviews_df.shape)
old_df = pd.read_csv('resources//data//doc2vec.csv')
old_df = old_df.astype(str)
tfidf = TfidfVectorizer(min_df=10,stop_words='english')
test = tfidf.fit_transform(old_df["review_clean"]).toarray()
tfidf_result = tfidf.transform(reviews_df["review_clean"]).toarray()
tfidf_df = pd.DataFrame(tfidf_result, columns = tfidf.get_feature_names())
tfidf_df.columns = ["word_" + str(x) for x in tfidf_df.columns]
tfidf_df.index = reviews_df.index
reviews_df = pd.concat([reviews_df, tfidf_df], axis=1)
print(reviews_df.shape)
print(tfidf_df.shape)
reviews_df.to_csv('resources//data//test_generic_tfidf2.csv')
old_df1 = pd.read_csv('resources//data//tfidf.csv')

print('--------pred-----------')
print(reviews_df.shape)
print('----------train---------')
print(old_df1.shape)

train = old_df1.columns
test = reviews_df.columns
columns_to_remove = [i for i in test if i not in train]
columns_to_add = [i for i in train if i not in test]
print('removing columns {}'.format(columns_to_remove))
reviews_df.drop(columns_to_remove,axis=1,inplace=True)
print('adding columns {}'.format(columns_to_add))
for i in columns_to_add:
    reviews_df[i] = 0
    
print('------reindexing to test columns-----')
reviews_df = reviews_df.reindex(columns=train)

print('--------train-----------')
print(columns_to_remove)
print(len(columns_to_remove))
print('----------pred---------')
print(columns_to_add)
print(len(columns_to_add))

print('--------pred-----------')
print(reviews_df.shape)
print('----------train---------')
print(old_df1.shape)

print('-----------feature selection-----------')
label = "is_bad_review"
ignore_cols = [label, "review", "review_clean"]
features = [c for c in reviews_df.columns if c not in ignore_cols]

# split the data into train and test
X_test = reviews_df[features]
y_test = reviews_df[label]

print(X_test.shape)
print(y_test.shape)

print('-------load a random forest classifier----------')
with open('resources//pickle//rf.pkl','rb') as rp:
    rf = pickle.load(rp)

print('--------------Predition accuracy---------')
y_pred = rf.predict(X_test)
print('F1-score : {}'.format(rf.score(X_test,y_test)))

print('----------Matrix--------------')
print(confusion_matrix(y_test,y_pred))  
print(classification_report(y_test,y_pred))  
print(accuracy_score(y_test, y_pred))

