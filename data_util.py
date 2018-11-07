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

stopWords = set(stopwords.words('english'))
stemmer = SnowballStemmer('english')

def load_data():
    oe = json.loads(open('data/OpenEnded_mscoco_val2014_questions.json','r').read())
    anno = json.loads(open('data/mscoco_val2014_annotations.json','r').read())
    
    return oe, anno
    
def create_question_vocab(question_json):
    vocab = defaultdict(int)
    questions = []

    for question in question_json['questions']:
        question = question['question']
        tokens = word_tokenize(question)
        filterd_tokens = []
        
        for word in tokens:
            if word.isalpha() and word not in stopWords:
                word = stemmer.stem(word.lower())
                vocab[word] += 1
                filterd_tokens.append(word)
                
        questions.append(filterd_tokens) 
                
    vocab = list(vocab.items())
    vocab.sort(key=itemgetter(1), reverse=True)     

    return dict(vocab[0:1000]), questions

def drop_words(vocab, questions):
    mod_questions = []
    vocab_list = vocab.keys()
    for q in questions:
        words = []
        for word in q:
            if word in vocab_list:
                words.append(word)
        mod_questions.append(words)
        
    return mod_questions

def one_hot_encode(vocab, questions):
    vect = np.zeros((len(questions), 1000))
    wdict = {token: idx for idx, token in enumerate(vocab.keys())}

    for idx, q in enumerate(questions):
        for word in q:
            vect[idx][wdict[word]] += 1
            
    return vect