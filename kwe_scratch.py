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
                trigram_dict[key][key1][key2] /= count

                    

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
                    quadgram_dict[key][key1][key2][key3] /= count

                    

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
                        pentagram_dict[key][key1][key2][key3][key4] /= count

    return pentagram_dict

def evaluate(sentence, label, penta_dict, quad_dict, tri_dict,bi_dict, uni_dict):
    
    for i in range(len(sentence)-4):
        prob = 1
        if i == 0:
            prob = uni_dict[sentence[i]]
        elif i == 1:
            prob *= bi_dict[sentence[i]][sentence[i+1]]
        elif i == 2:
            prob *= tri_dict[sentence[i]][sentence[i+1]][sentence[i+2]]
        elif i == 3:
            prob *= quad_dict[sentence[i]][sentence[i+1]][sentence[i+2]][sentence[i+3]]
        else:
            prob *= penta_dict[sentence[i]][sentence[i+1]][sentence[i+2]][sentence[i+3]][sentence[i+4]]
        
    prob *= penta_dict[sentence[i+1]][sentence[i+2]][sentence[i+3]][sentence[i+4]][label]
    
    if prob == 0:
        prob = 1
        
    return prob
            
if __name__ == "__main__":
    name = 'cwe-train.txt'
    corpus = load_dataset(name)
    
    smoothing_factor = 0.1


    count_dict, prob_dict = get_char_prob(corpus, smoothing_factor)
    bigram_probs = get_bigram_prob(count_dict, corpus, smoothing_factor)
    trigram_probs = get_trigram_prob(bigram_probs, count_dict, corpus, smoothing_factor)
    quadgram_probs = get_quadgram_prob(count_dict, corpus, smoothing_factor)
    pentagram_probs = get_pentagram_prob(count_dict, corpus, smoothing_factor)
    df = pd.DataFrame(data=bigram_probs)
    
    history = 10
    chars_seq = create_sequences(corpus, history)
    chars_seq = np.array(chars_seq)
    
    X = []
    y = []

    for i in range(len(chars_seq)):
        X.append(chars_seq[i][:-1])
        y.append(chars_seq[i][-1])
        
    loss = 0
    
    for i in range(len(X)):
        loss -= log2(evaluate(X[i], y[i], pentagram_probs, quadgram_probs, trigram_probs, bigram_probs, prob_dict))
        
    cross_entropy = loss / len(X)

    # predict = []

    # for sentence in X:
    #     probs = dict.fromkeys(count_dict.keys())
    #     for key in probs.keys():
    #         prob = 0
    #         for i in range(len(sentence)-1):
    #             if i == 0:
    #                 prob = prob_dict[sentence[i]]
    #             else:
    #                 prob *= bigram_probs[sentence[i]][sentence[i+1]]
    #             probs[key] = bigram_probs[sentence[i+1]][key] * prob
    #     predict.append(max(probs, key=probs.get))

    # y_ = y[:len(predict)]

    # correct = 0
    # for i in range(len(predict)):
    #     if predict[i] == y_[i]:
    #         correct += 1

    # accuracy = correct/len(predict)


#bible_test = 45*nltk.sent_tokenize(bible1)
#bible_test1 = nltk.sent_tokenize(bible1)
#for i in range(len(bible_test)):
#    bible_test [i] = re.sub(r'\W',' ',bible_test [i])
#    bible_test [i] = re.sub(r'\s+',' ',bible_test [i])
#    bible_test [i] = unicodedata.normalize('NFC', bible_test [i])
#    bible_test [i] = bible_test [i].lower()
#
#bible_test_dict = {}
#for sentence in bible_test:
#    tokens = nltk.word_tokenize(sentence)
#    for token in tokens:
#        irish_vowels = ["A","E","I","O","U","\u00C1","\u00C9","\u00CD","\u00D3","\u00DA",
#                    "a","e","i","o","u","á","é","í","ó","ú"]
#        if len(token) != 1:
#            if (token[0] == 'n' or token[0] == 't') and token[1] in irish_vowels and  len(token) != 2:
#                token = token[0] + '-' + token[1:]
#        if token not in bible_test_dict.keys():
#            bible_test_dict[token] = 1
#        else:
#            bible_test_dict[token] += 1

# for i in range(len(corpus)):
#     corpus [i] = re.sub(r'\W',' ',corpus [i])
#     corpus [i] = re.sub(r'\s+',' ',corpus [i])
#     corpus [i] = unicodedata.normalize('NFC', corpus [i])
#     corpus [i] = corpus [i].lower()

# irish_dict = {}
# for sentence in corpus:
#     tokens = nltk.word_tokenize(sentence)
#     for token in tokens:
#         irish_vowels = ["A","E","I","O","U","\u00C1","\u00C9","\u00CD","\u00D3","\u00DA",
#                     "a","e","i","o","u","á","é","í","ó","ú"]
#         if len(token) != 1:
#             if (token[0] == 'n' or token[0] == 't') and token[1] in irish_vowels and  len(token) != 2:
#                 token = token[0] + '-' + token[1:]
#         if token not in irish_dict.keys():
#             irish_dict[token] = 1
#         else:
#             irish_dict[token] += 1

# for key in irish_dict.copy():
#     if irish_dict[key] < 45:
#         irish_dict.pop(key)

# numbers = '''0123456789_'''
# for key in irish_dict.copy():
#     for element in key:
#        if element in numbers:
#            irish_dict.pop(key)
#            break

# irish_dict = {k: v for k, v in sorted(irish_dict.items(), reverse = True, key=lambda item: item[1])}

# total = 0
# for key in irish_dict:
#     total += irish_dict[key]

# frequency_dict = {}
# for key in irish_dict:
#     frequency_dict[key] = irish_dict[key]/total


# file = open("irish_dictionary.txt", "w", encoding = "UTF-16")
# file.write("%s\n" % (irish_dict))
# file.close()

# file = open("irish_frequency.txt", "w", encoding = "UTF-16")
# file.write("%s\n" % (frequency_dict))
# file.close()

# import ast
# file = open("irish_dictionary.txt", "r", encoding = "UTF-16")
# corpus1 = file.read()
# irish_dictionary = ast.literal_eval(corpus)
# file.close()

# file = open("irish_frequency.txt", "r", encoding = "UTF-16")
# corpus1 = file.read()
# frequency_dictionary = ast.literal_eval(corpus)
# file.close()

# file = open("irish_dictionary_old.txt", "r", encoding = "UTF-16")
# corpus1 = file.read()
# irish_dictionary_old = ast.literal_eval(corpus)
# file.close()

# file = open("irish_frequency_old.txt", "r", encoding = "UTF-16")
# corpus1 = file.read()
# frequency_dictionary_old = ast.literal_eval(corpus)
# file.close()

# from nltk import ngrams

# n = 3
# grams_irish_dict_3 = {}
# for sentence in corpus:
#     grams= ngrams(nltk.word_tokenize(sentence), n)
#     for gram in grams:
#         if gram not in grams_irish_dict_3.keys():
#             grams_irish_dict_3[gram] = 1
#         else:
#             grams_irish_dict_3[gram] += 1

# grams_irish_dict_3 = {k: v for k, v in sorted(grams_irish_dict_3.items(), reverse = True, key=lambda item: item[1])}

# for key in grams_irish_dict_3.copy():
#     if grams_irish_dict_3[key] < 15:
#         grams_irish_dict_3.pop(key)

# total = 0
# for key in grams_irish_dict_3:
#     total += grams_irish_dict_3[key]

# gram_frequency_dict_3 = {}
# for key in grams_irish_dict_3:
#     gram_frequency_dict_3[key] = grams_irish_dict_3[key]/total

# file = open("gram_irish_frequency_3a.txt", "w", encoding = "UTF-16")
# file.write("%s\n" % (gram_frequency_dict_3))
# file.close()

# import ast
# file = open("gram_irish_dictionary.txt", "r", encoding = "UTF-16")
# corpus1 = file.read()
# gram_irish_dictionary = ast.literal_eval(corpus1)
# file.close()


# import ast
# file1 = open("irish_dictionary.txt", "r", encoding = "UTF-16")
# corpus1 = file1.read()
# irish_dictionary = ast.literal_eval(corpus1)
# file1.close()

# irish_alphabet = '''abcdefghilmnoprstuáéíóú-'''
# for key in irish_dictionary.copy():
#     for element in key:
#        if element not in irish_alphabet:
#            irish_dictionary.pop(key)
#            break

# file2 = open("irish_dictionary_filtered.txt", "w", encoding = "UTF-16")
# file2.write("%s\n" % (irish_dictionary))
# file2.close()

# file3 = open("gram_irish_dictionary_3a.txt", "r", encoding = "UTF-16")
# corpus2 = file3.read()
# gram_irish_dictionary_3a = ast.literal_eval(corpus2)
# file3.close()

# irish_alphabet = '''abcdefghilmnoprstuáéíóú-'''
# for key in gram_irish_dictionary_3a.copy():
#     break_flag = False
#     for element in key:
#         if break_flag:
#             break
#         for letter in element:
#             if letter not in irish_alphabet:
#                 gram_irish_dictionary_3a.pop(key)
#                 break_flag = True
#                 break

# file4 = open("gram_irish_dictionary_3a_filtered.txt", "w", encoding = "UTF-16")
# file4.write("%s\n" % (gram_irish_dictionary_3a))
# file4.close()

# total = 0
# for key in gram_irish_dictionary_3a:
#     total += gram_irish_dictionary_3a[key]

# gram_frequency_dict_3a = {}
# for key in gram_irish_dictionary_3a:
#     gram_frequency_dict_3a[key] = gram_irish_dictionary_3a[key]/total

# file5 = open("gram_irish_frequency_filtered_3a.txt", "w", encoding = "UTF-16")
# file5.write("%s\n" % (gram_frequency_dict_3a))
# file5.close()


# file6 = open("irish_dictionary_filtered.txt", "r", encoding = "UTF-16")
# corpus3 = file6.read()
# irish_dictionary1 = ast.literal_eval(corpus3)
# file6.close()

# total = 0
# for key in irish_dictionary1:
#     total += irish_dictionary1[key]

# frequency_dict = {}
# for key in irish_dictionary1:
#     frequency_dict[key] = irish_dictionary1[key]/total

# file7 = open("irish_frequency_filtered.txt", "w", encoding = "UTF-16")
# file7.write("%s\n" % (frequency_dict))
# file7.close()


