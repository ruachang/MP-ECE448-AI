# naive_bayes.py
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
util for printing values
'''
def print_values(laplace, pos_prior):
    print(f"Unigram Laplace: {laplace}")
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

"""
Count all the words in the train set and return the conditonal prob of
each word in each kind
Use the type count to help simplify the process
"""
def count_word(train_set, train_labels):
    type_num = Counter({1 : 0, 0 : 0})   
    positive_list = Counter()
    negetive_list = Counter()
    # type_num.update(train_labels)
    pos_num = 0
    for i in range(len(train_labels)):
        # type_num.update(train_labels[i])
        if train_labels[i] == 1:
            positive_list.update(train_set[i])
            type_num[1] += len(train_set[i])
            pos_num += 1
        else:
            negetive_list.update(train_set[i])
            type_num[0] += len(train_set[i])
        prior_prob = pos_num / len(train_labels)
    return type_num, positive_list, negetive_list, prior_prob

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
    # positive_prob.most_common()
    # negetive_prob.most_common()
    # return positive_prob, negetive_prob
    
def make_prediction(word, alpha, type_num, positive_list, negetive_list, prior_prob):
    pos_probs = math.log(prior_prob)
    neg_probs = math.log(1 - prior_prob)
    for i in word:
        pos_prob, neg_prob = prob_calculate(i, alpha, type_num, positive_list, negetive_list)
        pos_probs += pos_prob
        neg_probs += neg_prob
    if neg_probs > pos_probs:
        return 0
    else:
        return 1

"""
Main function for training and predicting with naive bayes.
    You can modify the default values for the Laplace smoothing parameter and the prior for the positive label.
    Notice that we may pass in specific values for these parameters during our testing.
"""
def naiveBayes(dev_set, train_set, train_labels, laplace=0.07, pos_prior=0.5, silently=False):
    print_values(laplace,pos_prior)

    type_num, positive_list, negetive_list, _ = count_word(train_set, train_labels)
    
    # positive_prob, negetive_prob = prob_calculate(type_num, positive_list, negetive_list)
    yhats = []
    for doc in tqdm(dev_set, disable=silently):
        yhats.append(make_prediction(doc, laplace, type_num, positive_list, negetive_list, pos_prior))
    
    return yhats
