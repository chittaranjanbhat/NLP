# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 14:34:16 2019

@author: chitt
"""

import logging
from logging import config
import pandas as pd
import string
import pickle
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

config.fileConfig(fname='config\\logger.config', disable_existing_loggers=False)
logger = logging.getLogger(__name__)

class TrainPipeline:
    def __init__(self,input_data):
        logger.info("Training Initited on data {}".format(input_data))
        self.df = pd.read_csv(input_data)
        
    def train_preprocess(self):
        logger.info("Shape of data set {}".format(self.df.shape))
        # clean text data
        logger.info("----clean text data-------")
        self.df["review_clean"] = self.df["review"].apply(lambda x: clean_text(x))
        
        logger.info("--------add sentiment anaylsis columns--------")
        sid = SentimentIntensityAnalyzer()
        self.df["sentiments"] = self.df["review"].apply(lambda x: sid.polarity_scores(x))
        self.df = pd.concat([self.df.drop(['sentiments'], axis=1), self.df['sentiments'].apply(pd.Series)], axis=1)
        
        logger.info("-------add number of characters column-------")
        self.df["nb_chars"] = self.df["review"].apply(lambda x: len(x))

        logger.info("------ add number of words column--------")
        self.df["nb_words"] = self.df["review"].apply(lambda x: len(x.split(" ")))
        
        logger.info("-------- create doc2vec vector columns------")
        documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(self.df["review_clean"].apply(lambda x: x.split(" ")))]
        
        logger.info("-------- train a Doc2Vec model with our text data----")
        model = Doc2Vec(documents, vector_size=5, window=2, min_count=1, workers=4)
        
        logger.info("------transform each document into a vector data------")
        doc2vec_df = self.df["review_clean"].apply(lambda x: model.infer_vector(x.split(" "))).apply(pd.Series)
        doc2vec_df.columns = ["doc2vec_vector_" + str(x) for x in doc2vec_df.columns]
        self.df = pd.concat([self.df, doc2vec_df], axis=1)
        
        
        
        
        
        
        
        
        
        
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
        
        
        # return the wordnet object value corresponding to the POS tag

