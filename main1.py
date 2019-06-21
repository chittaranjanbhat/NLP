# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 00:19:17 2019

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


# read data
if not path.exists("resources//data//clean.csv"):
    df = pd.read_csv("resources//data//Hotel_Reviews.csv")
    # append the positive and negative text reviews
    df["review"] = df["Negative_Review"] + df["Positive_Review"]
    # create the label
    df["is_bad_review"] = df["Reviewer_Score"].apply(lambda x: -1 if x <=4 else 0 if x<=7 else 1)
    # select only relevant columns
    df = df[["review", "is_bad_review"]]
    df_negative = df[df.is_bad_review ==-1]
    df_positive = df[df.is_bad_review ==1]
    df_neutral = df[df.is_bad_review ==0]
    
    #Shuffling the positive and neutral records
    df_positive.reindex(np.random.permutation(df_positive.index))
    df_neutral.reindex(np.random.permutation(df_neutral.index))
    
    #Balancing the negative,positive and neutral document
    df_positive = df_positive[:len(df_negative)]
    df_neutral = df_neutral[:len(df_negative)]
    
    #combine all negative,positive and neutral document
    reviews_df = pd.concat([df_positive,df_neutral,df_negative], ignore_index=True)
    
    #Check the balancing of labels
    print(reviews_df.groupby('is_bad_review').count())
    # clean text data
    reviews_df["review_clean"] = reviews_df["review"].apply(lambda x: clean_text(x))

    print(reviews_df.shape)
    print(reviews_df.head()) 
    reviews_df.to_csv("resources//data//clean.csv")
    print('Clean file saved')
else:
    print('Clean file present')
    reviews_df = pd.read_csv("resources//data//clean.csv")
    reviews_df = reviews_df.sample(frac = 0.1, replace = False, random_state=42)
    print(reviews_df.shape)
    
print("--------------Feature engineering----------")
if not path.exists('resources//data//SentimentIntensityAnalyzer.csv'):
    reviews_df.reindex(np.random.permutation(reviews_df.index))
    sid = SentimentIntensityAnalyzer()
    reviews_df["sentiments"] = reviews_df["review"].apply(lambda x: sid.polarity_scores(x))
    reviews_df = pd.concat([reviews_df.drop(['sentiments'], axis=1), reviews_df['sentiments'].apply(pd.Series)], axis=1)
    
    print(reviews_df.shape)
    
    # add number of characters column
    reviews_df["nb_chars"] = reviews_df["review"].apply(lambda x: len(x))
    # add number of words column
    reviews_df["nb_words"] = reviews_df["review"].apply(lambda x: len(x.split(" ")))
    
    print(reviews_df.shape)
    reviews_df["review_clean"] = reviews_df["review_clean"].astype(str)
    reviews_df.to_csv('resources//data//SentimentIntensityAnalyzer.csv')
print("---add metrics for every text-------")
if not path.exists('resources//pickle//Doc2Vec_model.pkl'):
    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(reviews_df["review_clean"].apply(lambda x: x.split(" ")))]
    # train a Doc2Vec model with our text data
    model = Doc2Vec(documents, vector_size=5, window=2, min_count=1, workers=4)
    with open('resources//pickle//Doc2Vec_model.pkl','wb') as pf:
        pickle.dump(model,pf)

# transform each document into a vector data
if not path.exists('resources//data//doc2vec.csv'):
    doc2vec_df = reviews_df["review_clean"].apply(lambda x: model.infer_vector(x.split(" "))).apply(pd.Series)
    doc2vec_df.columns = ["doc2vec_vector_" + str(x) for x in doc2vec_df.columns]
    reviews_df = pd.concat([reviews_df, doc2vec_df], axis=1)
    print(reviews_df.shape)
    reviews_df.to_csv('resources//data//doc2vec.csv')

# add tf-idfs columns
if not path.exists('resources//data//tfidf.csv'):
    tfidf = TfidfVectorizer(min_df = 10)
    tfidf_result = tfidf.fit_transform(reviews_df["review_clean"]).toarray()
    tfidf_df = pd.DataFrame(tfidf_result, columns = tfidf.get_feature_names())
    tfidf_df.columns = ["word_" + str(x) for x in tfidf_df.columns]
    tfidf_df.index = reviews_df.index
    reviews_df = pd.concat([reviews_df, tfidf_df], axis=1)
    print(reviews_df.shape)
    print(tfidf_df.shape)
    with open('resources//pickle//tfidf_result.pkl','wb') as pf:
        pickle.dump(tfidf_result,pf)
    
    reviews_df.to_csv('resources//data//tfidf.csv')
    print(tfidf_result)
    with open('resources//pickle//tfidf_result.pkl','wb') as pf:
        pickle.dump(tfidf_result,pf)
         
else:
    reviews_df = pd.read_csv('resources//data//tfidf.csv')


print(reviews_df.shape)

print('-----------feature selection-----------')
label = "is_bad_review"
ignore_cols = [label, "review", "review_clean"]
features = [c for c in reviews_df.columns if c not in ignore_cols]

# split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(reviews_df[features], reviews_df[label], test_size = 0.20, random_state = 42)

print(X_train.shape)
print(X_test.shape)

print('-------train a random forest classifier----------')
rf = RandomForestClassifier(n_estimators = 100, random_state = 42)
rf.fit(X_train, y_train)
with open('resources//pickle//rf.pkl','wb') as pf:
    pickle.dump(rf,pf)

# show feature importance
feature_importances_df = pd.DataFrame({"feature": features, "importance": rf.feature_importances_}).sort_values("importance", ascending = False)
print(feature_importances_df.head(20))

print('--------------Testing the accuracy---------')
y_pred = rf.predict(X_test)
print('F1-score : {}'.format(rf.score(X_test,y_test)))

print('----------Matrix--------------')
print(confusion_matrix(y_test,y_pred))  
print(classification_report(y_test,y_pred))  
print(accuracy_score(y_test, y_pred))


def get_wordnet_pos(pos_tag):
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def clean_text(text):
    # lower text
    text = text.lower()
    # tokenize text and remove puncutation
    text = [word.strip(string.punctuation) for word in text.split(" ")]
    # remove words that contain numbers
    text = [word for word in text if not any(c.isdigit() for c in word)]
    # remove stop words
    stop = stopwords.words('english')
    text = [x for x in text if x not in stop]
    # remove empty tokens
    text = [t for t in text if len(t) > 0]
    # pos tag text
    pos_tags = pos_tag(text)
    # lemmatize text
    text = [WordNetLemmatizer().lemmatize(t[0], get_wordnet_pos(t[1])) for t in pos_tags]
    # remove words with only one letter
    text = [t for t in text if len(t) > 1]
    # join all
    text = " ".join(text)
    return(text)