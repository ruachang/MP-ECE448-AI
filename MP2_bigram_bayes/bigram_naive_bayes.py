# bigram_naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
# Last Modified 8/23/2023


"""
This is the main code for this MP.
You only need (and should) modify code within this file.
Original staff versions of all other files will be used by the autograder
so be careful to not modify anything else.
"""


import reader
import math
from tqdm import tqdm
from collections import Counter


'''
utils for printing values
'''
def print_values(laplace, pos_prior):
    print(f"Unigram Laplace: {laplace}")
    print(f"Positive prior: {pos_prior}")

def print_values_bigram(unigram_laplace, bigram_laplace, bigram_lambda, pos_prior):
    print(f"Unigram Laplace: {unigram_laplace}")
    print(f"Bigram Laplace: {bigram_laplace}")
    print(f"Bigram Lambda: {bigram_lambda}")
    print(f"Positive prior: {pos_prior}")
"""
load_data loads the input data by calling the provided utility.
You can adjust default values for stemming and lowercase, when we haven't passed in specific values,
to potentially improve performance.
"""
def load_data(trainingdir, testdir, stemming=False, lowercase=False, silently=False):
    print(f"Stemming: {stemming}")
    print(f"Lowercase: {lowercase}")
    train_set, train_labels, dev_set, dev_labels = reader.load_dataset(trainingdir,testdir,stemming,lowercase,silently)
    return train_set, train_labels, dev_set, dev_labels

def pair_data(set):
    pair_set = []
    for sentence in set:
        pair_words = []
        for word in range(len(sentence) - 1):
            pair_word = sentence[word] + sentence[word + 1]
            pair_words.append(pair_word)
        pair_set.append(pair_words)
    return pair_set

"""
Count all the words in the train set and return the conditonal prob of
each word in each kind
Use the type count to help simplify the process
"""
def count_word(train_set, train_labels):
    type_num = Counter({1 : 0, 0 : 0})   
    positive_list = Counter()
    negetive_list = Counter()
    positive_set = Counter()
    negetive_set_list = Counter()
    set_num = Counter({1 : 0, 0 : 0})
    # type_num.update(train_labels)
    for i in range(len(train_labels)):
        # type_num.update(train_labels[i])
        if train_labels[i] == 1:
            positive_list.update(train_set[i])
            type_num[1] += len(train_set[i])
        else:
            negetive_list.update(train_set[i])
            type_num[0] += len(train_set[i])

    return type_num, positive_list, negetive_list

def count_set_word(train_set, train_labels):
    positive_set_list = Counter()
    negetive_set_list = Counter()
    set_num = Counter({1 : 0, 0 : 0})
    pair_set = pair_data(train_set)
    # type_num.update(train_labels)
    for i in range(len(train_labels)):
        # type_num.update(train_labels[i])
        if train_labels[i] == 1:
            positive_set_list.update(pair_set[i])
            set_num[1] += len(pair_set[i])
        else:
            negetive_set_list.update(pair_set[i])
            set_num[0] += len(pair_set[i])
    return set_num, positive_set_list, negetive_set_list

def laplace_smooth(alpha, key, num, word_list):
    kind = len(word_list.keys())
    if key in word_list.keys():
        prob = (word_list[key] + alpha) / (num + (kind + 1) * alpha)
    else:
        prob = alpha / (num + (kind + 1) * alpha)
    return math.log(prob)

def prob_calculate(word, alpha, type_num, positive_list, negetive_list):
    # positive_prob = Counter()
    # for key in positive_list.keys():
        # positive_prob[keys] = positive_list[keys] / type_num[1]
    pos_prob = laplace_smooth(alpha, word, type_num[1], positive_list)
    # negetive_prob = Counter()
    # for keys in negetive_list.keys():
    neg_prob = laplace_smooth(alpha, word, type_num[0], negetive_list)

    return pos_prob, neg_prob

def make_prediction(word, unigram_alpha, bigram_alpha, lamda, 
                    type_num, positive_list, negetive_list, 
                    set_num, positive_set_list, negetive_set_list, 
                    prior_prob):
    pos_probs = math.log(prior_prob)
    neg_probs = math.log(1 - prior_prob)
    word_pos_probs = 0
    word_neg_probs = 0
    word_set = pair_data([word])[0]
    for i in word:
        pos_prob, neg_prob = prob_calculate(i, unigram_alpha, type_num, positive_list, negetive_list)
        word_pos_probs += pos_prob
        word_neg_probs += neg_prob
    wordset_pos_probs = 0
    wordset_neg_probs = 0
    for i in word_set:
        pos_prob, neg_prob = prob_calculate(i, bigram_alpha, set_num, positive_set_list, negetive_set_list)
        wordset_pos_probs += pos_prob
        wordset_neg_probs += neg_prob
    pos_probs = (1 - lamda) * word_pos_probs + lamda * wordset_pos_probs
    neg_probs = (1 - lamda) * word_neg_probs + lamda * wordset_neg_probs
        
    if neg_probs > pos_probs:
        return 0
    else:
        return 1
"""
Main function for training and predicting with the bigram mixture model.
    You can modify the default values for the Laplace smoothing parameters, model-mixture lambda parameter, and the prior for the positive label.
    Notice that we may pass in specific values for these parameters during our testing.
"""
def bigramBayes(dev_set, train_set, train_labels, unigram_laplace=0.005, bigram_laplace=0.005, bigram_lambda=0.5, pos_prior=0.5, silently=False):
    print_values_bigram(unigram_laplace,bigram_laplace,bigram_lambda,pos_prior)

    type_num, positive_list, negetive_list = count_word(train_set, train_labels)
    set_num, positive_set_list, negetive_set_list = count_set_word(train_set, train_labels)
    yhats = []
    for doc in tqdm(dev_set, disable=silently):
        yhats.append(make_prediction(doc, unigram_laplace, bigram_laplace, bigram_lambda,
                                     type_num, positive_list, negetive_list, 
                                     set_num, positive_set_list, negetive_set_list, pos_prior
                                     ))

    return yhats
