# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 12:11:19 2022

@author: omars
"""

import pickle
import numpy as np
import pandas as pd
from math import log2


def load_dataset(file_name):
    with open(file_name, encoding= "UTF-8") as file:
        corpus = file.read()
    file.close()
    
    ncorpus = corpus.replace("\n", " " )

    return ncorpus

def get_char_prob(corpus,smoothing_factor):
    count_dict = {}
    for sentence in corpus:
        for char in sentence:
            if char in count_dict.keys():
                count_dict[char] += 1
            else:
                count_dict[char] = 1 + smoothing_factor

    count_dict = {k: v for k, v in count_dict.items() if v!=1}

    char_num = sum(count_dict.values())

    prob_dict = {}

    for key in count_dict.keys():
        prob_dict[key] = count_dict[key] /char_num

    return count_dict, prob_dict

def create_sequences(corpus, history):
    tokens = corpus.split()
    corpus = ' '.join(tokens)
    sequences = list()
    for i in range(history, len(corpus)):
    	seq = corpus[i-history:i+1]
    	sequences.append(seq)
    print('Total Sequences: %d' % len(sequences))
    return sequences

def get_bigram_prob(count_dict, corpus, smoothing_factor):
    smoothing_factor = 1
    bigram_dict = {}
    for key in count_dict.keys():
        bigram_dict[key] = {}
        for key1 in count_dict.keys():
            bigram_dict[key][key1] = smoothing_factor

    for i in range(len(corpus)-1):
            bigram_dict[corpus[i]][corpus[i+1]] += 1

    for key in bigram_dict.keys():
        count = sum(bigram_dict[key].values())
        for key1 in bigram_dict[key].keys():
            bigram_dict[key][key1] /= count

    return bigram_dict

def get_trigram_prob(bigram_dict, count_dict, corpus, smoothing_factor):
    trigram_dict = {}
    for key in count_dict.keys():
        trigram_dict[key] = {}
        for key1 in count_dict.keys():
            trigram_dict[key][key1] = {}
            for key2 in count_dict.keys():
                trigram_dict[key][key1][key2] = smoothing_factor

    for i in range(len(corpus)-2):
        trigram_dict[corpus[i]][corpus[i+1]][corpus[i+2]] += 1

    for key in trigram_dict.keys():
        for key1 in trigram_dict[key].keys():
            count = sum(trigram_dict[key][key1].values())
            for key2 in trigram_dict[key][key1].keys():
                try:
                    trigram_dict[key][key1][key2] /= count
                except:
                    trigram_dict[key][key1][key2] = 0 
                    
    return trigram_dict

def get_quadgram_prob(count_dict, corpus, smoothing_factor):
    quadgram_dict = {}
    for key in count_dict.keys():
        quadgram_dict[key] = {}
        for key1 in count_dict.keys():
            quadgram_dict[key][key1] = {}
            for key2 in count_dict.keys():
                quadgram_dict[key][key1][key2] = {}
                for key3 in count_dict.keys():
                    quadgram_dict[key][key1][key2][key3] = smoothing_factor

    for i in range(len(corpus)-3):
        quadgram_dict[corpus[i]][corpus[i+1]][corpus[i+2]][corpus[i+3]] += 1

    for key in quadgram_dict.keys():
        for key1 in quadgram_dict[key].keys():
            for key2 in quadgram_dict[key][key1].keys():
                count = sum(quadgram_dict[key][key1][key2].values())
                for key3 in quadgram_dict[key][key1][key2].keys():
                    try:
                        quadgram_dict[key][key1][key2][key3] /= count
                    except:
                        quadgram_dict[key][key1][key2][key3] = 0                    

    return quadgram_dict

def get_pentagram_prob(count_dict, corpus, smoothing_factor):
    pentagram_dict = {}
    for key in count_dict.keys():
        pentagram_dict[key] = {}
        for key1 in count_dict.keys():
            pentagram_dict[key][key1] = {}
            for key2 in count_dict.keys():
                pentagram_dict[key][key1][key2] = {}
                for key3 in count_dict.keys():
                    pentagram_dict[key][key1][key2][key3] = {}
                    for key4 in count_dict.keys():
                        pentagram_dict[key][key1][key2][key3][key4] = smoothing_factor

    for i in range(len(corpus)-4):
        pentagram_dict[corpus[i]][corpus[i+1]][corpus[i+2]][corpus[i+3]][corpus[i+4]] += 1

    for key in pentagram_dict.keys():
        for key1 in pentagram_dict[key].keys():
            for key2 in pentagram_dict[key][key1].keys():
                for key3 in pentagram_dict[key][key1][key2].keys():
                    count = sum(pentagram_dict[key][key1][key2][key3].values())
                    for key4 in pentagram_dict[key][key1][key2][key3].keys():
                        try:
                            pentagram_dict[key][key1][key2][key3][key4] /= count
                        except:
                            pentagram_dict[key][key1][key2][key3][key4] = 0 
                            
    return pentagram_dict

def smooth_grams(penta_grams,quad_grams,tri_grams,bi_grams,uni_grams,vocab):
    L1 = 0.035
    L2 = 0.09
    L3 = 0.125
    L4 = 0.25
    L5 = 0.50
    smoothed = {}
    for key in vocab:
        smoothed[key] = {}
        for key1 in vocab:
            smoothed[key][key1] = {}
            for key2 in vocab:
                smoothed[key][key1][key2] = {}
                for key3 in vocab:
                    smoothed[key][key1][key2][key3] = {}
                    for key4 in vocab:
                        smoothed[key][key1][key2][key3][key4] = (L1 * uni_grams[key4]) + (L2 * bi_grams[key3][key4]) + (L3 * tri_grams[key2][key3][key4]) + (L4 * quad_grams[key1][key2][key3][key4]) + (L5 * penta_grams[key][key1][key2][key3][key4])
                        
    return smoothed

def evaluate(sentence, label, penta_dict):
    prob = 1
    for i in range(4,len(sentence)-4):        
        # if i == 0:
        #     prob = uni_dict[sentence[i]]
        # elif i == 1:
        #     prob *= bi_dict[sentence[i]][sentence[i+1]]
        # elif i == 2:
        #     prob *= tri_dict[sentence[i]][sentence[i+1]][sentence[i+2]]
        # elif i == 3:
        #     prob *= quad_dict[sentence[i]][sentence[i+1]][sentence[i+2]][sentence[i+3]]
        # else:
        prob *= penta_dict[sentence[i]][sentence[i+1]][sentence[i+2]][sentence[i+3]][sentence[i+4]]
        
    prob *= penta_dict[sentence[i+1]][sentence[i+2]][sentence[i+3]][sentence[i+4]][label]
    
    if prob == 0:
        prob = 1
        
    return prob
            
if __name__ == "__main__":
    name = 'cwe-train.txt'
    corpus = load_dataset(name)
    
    smoothing_factor = 0.1# add k-smoothing.


    count_dict, prob_dict = get_char_prob(corpus, smoothing_factor)
    vocab = count_dict.keys()
    bigram_probs = get_bigram_prob(count_dict, corpus, smoothing_factor)
    trigram_probs = get_trigram_prob(bigram_probs, count_dict, corpus, smoothing_factor)
    quadgram_probs = get_quadgram_prob(count_dict, corpus, smoothing_factor)
    pentagram_probs = get_pentagram_prob(count_dict, corpus, smoothing_factor)
    df = pd.DataFrame(data=bigram_probs) # visualization purposes
    
    history = 30
    chars_seq = create_sequences(corpus, history)
    chars_seq = np.array(chars_seq)
    
    X = []
    y = []

    for i in range(len(chars_seq)):
        X.append(chars_seq[i][:-1])
        y.append(chars_seq[i][-1])
    
    smoothed_probs = smooth_grams(pentagram_probs,quadgram_probs,trigram_probs,bigram_probs,prob_dict,vocab)
    
    loss = 0
    
    for i in range(len(X)):
        loss -= log2(evaluate(X[i], y[i], smoothed_probs))
        
    cross_entropy = loss / len(X)
