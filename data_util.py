# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 14:52:28 2018

@author: Michael
"""
import json
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from collections import defaultdict
from operator import itemgetter
from sklearn.preprocessing import OneHotEncoder

stopWords = set(stopwords.words('english'))
stemmer = SnowballStemmer('english')

def load_data():
    oe = json.loads(open('data/OpenEnded_mscoco_val2014_questions.json','r').read())
    anno = json.loads(open('data/mscoco_val2014_annotations.json','r').read())
    
    return oe, anno
    
def create_question_corpus(question_json):
    corpus = defaultdict(int)
    questions = []

    for question in question_json['questions']:
        question = question['question']
        tokens = word_tokenize(question)
        filterd_tokens = []
        
        for word in tokens:
            if word.isalpha() and word not in stopWords:
                word = stemmer.stem(word.lower())
                corpus[word] += 1
                filterd_tokens.append(word)
                
        questions.append(filterd_tokens) 
                
    corpus = list(corpus.items())
    corpus.sort(key=itemgetter(1), reverse=True)     

    return dict(corpus[0:1000]), questions

def drop_words(corpus, questions):
    mod_questions = []
    corpus_list = corpus.keys()
    for q in questions:
        words = []
        for word in q:
            if word in corpus_list:
                words.append(word)
        mod_questions.append(words)
        
    return mod_questions

def one_hot_encode(corpus, questions):
    word_to_id = {token: idx for idx, token in enumerate(corpus.keys())}
    tokens_docs = questions
    
    token_ids = [[word_to_id[token] for token in tokens_doc] for tokens_doc in tokens_docs]
    vec = OneHotEncoder(n_values=word_to_id.values())
    X = vec.fit_transform(np.asarray(token_ids[0]).reshape(1,-1))
    print(X.shape)
    