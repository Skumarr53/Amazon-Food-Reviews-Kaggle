import re
import string
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import nltk
from nltk.corpus import wordnet as wn, stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import unicodedata
from tqdm.notebook import tqdm
from pdb import set_trace


class Text_Summerizer(BaseEstimator, TransformerMixin):
    
    def fit(self,X,y=None):
        return self

    def transform(self,X):
        dic = {
            'count_letters' : lambda x: len(str(x)),
            'count_word' : lambda x: len(str(x).split()),
            'count_unique_word' : lambda x: len(set(str(x).split())),
            'count_sent' :  lambda x: len(nltk.sent_tokenize(str(x))),
            'count_punctuations': lambda x: len([c for c in str(x) if c in string.punctuation]),
            'mean_word_len' : lambda x: np.mean([len(w) for w in str(x).split()]),
            #'count_stopwords' : lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords])
        }        
        
        mat = np.zeros((len(X),len(dic)))
        
        
        
        for ind,col in enumerate(dic):
            mat[:,ind] =X.apply(dic[col]).values

        return pd.DataFrame(mat, columns = list(dic.keys()))


class Amaz_textCleaner(BaseEstimator, TransformerMixin):
    
    def __init__(self, stopwords=None):
        self.stopwords = stopwords
    
    def decontracted(self,phrase):
        #order should not change

        # specific
        phrase = re.sub(r"won't", "will not", phrase)
        phrase = re.sub(r"can\'t", "can not", phrase)

        # general
        phrase = re.sub(r"n\'t", " not", phrase)
        phrase = re.sub(r"\'re", " are", phrase)
        phrase = re.sub(r"\'s", " is", phrase)
        phrase = re.sub(r"\'d", " would", phrase)
        phrase = re.sub(r"\'ll", " will", phrase)
        phrase = re.sub(r"\'t", " not", phrase)
        phrase = re.sub(r"\'ve", " have", phrase)
        phrase = re.sub(r"\'m", " am", phrase)
        return phrase
    
    def fit(self, X, y=None):
        # stateless transformer
        return self
    
    def transform(self, X):
        preprocessed_reviews = []
        # tqdm is for printing the status bar
        for sentance in tqdm(X.values):
            sentance = sentance.lower() 
            sentance = re.sub(r"http\S+", "", sentance)
            sentance = BeautifulSoup(sentance, 'lxml').get_text()
            sentance = self.decontracted(sentance)
            sentance = re.sub("\S*\d\S*", "", sentance).strip()
            sentance = re.sub('[^A-Za-z]+', ' ', sentance)
            # https://gist.github.com/sebleier/554280
            sentance = ' '.join(e for e in sentance.split() if e not in self.stopwords)
            preprocessed_reviews.append(sentance.strip())
        return pd.Series(preprocessed_reviews)


class Avg_WV_transformer(BaseEstimator, TransformerMixin):

    def __init__(self,w2v_model = None):
        self.w2v_model = w2v_model

    def fit(self,X,y=None):
        return self

    def transform(self,X):
        list_of_sentance=[]
        for sentance in X:
            list_of_sentance.append(sentance.split())
        sent_vectors = []; # the avg-w2v for each sentence/review is stored in this list
        for sent in list_of_sentance: # for each review/sentence
            sent_vec = np.zeros(self.w2v_model.vector_size) # as word vectors are of zero length 50, you might need to change this to 300 if you use google's w2v
            cnt_words =0; # num of words with a valid vector in the sentence/review
            for word in sent: # for each word in a review/sentence
                if word in self.w2v_model.wv.vocab:
                    vec = self.w2v_model[word]
                    sent_vec += vec
                    cnt_words += 1
            if cnt_words != 0:
                sent_vec /= cnt_words
            sent_vectors.append(sent_vec)
        return np.stack(sent_vectors, axis=0)


class TFIDF_WV_transformer(BaseEstimator, TransformerMixin):
    
    def __init__(self,w2v_model = None, TFIDF_model = None):
        self.w2v_model = w2v_model
        self.TFIDF_model = TFIDF_model

    def fit(self,X,y=None):
        if self.TFIDF_model is None:  self.TFIDF_model = TfidfVectorizer()
        self.TFIDF_model.fit_transform(X)
        self.tfidf_dict = dict(zip(self.TFIDF_model.get_feature_names(), list(self.TFIDF_model.idf_)))
        return self

    def transform(self,X):
        list_of_sentance=[]
        tfidf_feat = self.TFIDF_model.get_feature_names()
        for sentance in X:
            list_of_sentance.append(sentance.split())
        
        tfidf_sent_vectors = []; # the tfidf-w2v for each sentence/review is stored in this list
        row=0;
        for sent in list_of_sentance: # for each review/sentence 
            sent_vec = np.zeros(self.w2v_model.vector_size) # as word vectors are of zero length
            weight_sum =0; # num of words with a valid vector in the sentence/review
            for word in sent: # for each word in a review/sentence
                if word in self.w2v_model.wv.vocab and word in tfidf_feat:
                    vec = self.w2v_model.wv[word]
        #             tf_idf = tf_idf_matrix[row, tfidf_feat.index(word)]
                    # to reduce the computation we are 
                    # dictionary[word] = idf value of word in whole courpus
                    # sent.count(word) = tf valeus of word in this review
                    tf_idf = self.tfidf_dict[word]*(sent.count(word)/len(sent))
                    sent_vec += (vec * tf_idf)
                    weight_sum += tf_idf
            if weight_sum != 0:
                sent_vec /= weight_sum
            tfidf_sent_vectors.append(sent_vec)
            row += 1
        return np.stack(tfidf_sent_vectors, axis=0)
