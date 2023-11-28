"""
Part 3: Here you should improve viterbi to use better laplace smoothing for unseen words
This should do better than baseline and your first implementation of viterbi, especially on unseen words
Most of the code in this file is the same as that in viterbi_1.py
"""

import math
from collections import defaultdict, Counter
from math import log

# Note: remember to use these two elements when you find a probability is 0 in the training data.
epsilon_for_pt = 1e-5
emit_epsilon = 1e-5   # exact setting seems to have little or no effect
suffix_list = set(["ment", "ly", "ous", "able", "less", "ness", "UNKNOWN"])
HAPAX_THRESH = 1

def prob_word_tag(word):
    sum_word_cnt = sum(word.values())
    for tag in word.keys():
        word[tag] = word[tag] / sum_word_cnt
        
# smooth the transmission probibility in collecting data
def smooth_transmission(trans_prob, unique_tag_list):
    for tag in trans_prob.keys():
        tag_dic = trans_prob[tag]
        num_processor = len(tag_dic.keys())
        lack_processor = len(unique_tag_list) - num_processor
        sum_word_cnt = sum(tag_dic.values()) + epsilon_for_pt * len(unique_tag_list)
        for processor_tag in unique_tag_list:
            if processor_tag not in tag_dic.keys():
                trans_prob[tag][processor_tag] = log(epsilon_for_pt * lack_processor) - log(sum_word_cnt)
            else:
                trans_prob[tag][processor_tag] = log(tag_dic[processor_tag]) - log(sum_word_cnt)
def get_hapax_word_set(word_cnt):
    hapax_set = []
    for word in word_cnt.keys():
        if word_cnt[word][0] < HAPAX_THRESH:
            hapax_set.append(word)
    return set(hapax_set)

def smooth_emit(emit_prob, word_cnt):
#     scale = 1000
    hapax_set = get_hapax_word_set(word_cnt)
    unknow_set = {}
    for tag in emit_prob.keys():
        num_minor = 1
        num_major = 0
        word_dic = emit_prob[tag]
        for word in word_dic.keys():
            if word in hapax_set:
                num_minor += 1
            else:
                num_major += 1
        scale = num_minor / len(hapax_set)
        sum_word_cnt = sum(word_dic.values()) + emit_epsilon * (len(word_dic.keys()) + 1)
        unknown_ratio = num_minor / len(word_dic.keys())
        # sum_word_cnt = sum(word_dic.values()) + \
        #         emit_epsilon * num_major + \
        #         emit_epsilon * scale * num_minor
        # for word in word_dic.keys():
        #     if word in hapax_set:
        #         emit_prob[tag][word] = log((word_dic[word] + scale * emit_epsilon)) - log(sum_word_cnt)
        #     else:
        #         emit_prob[tag][word] = log((word_dic[word] + emit_epsilon)) - log(sum_word_cnt)
        
        for word in word_dic.keys():
            if word in hapax_set:
                emit_prob[tag][word] = log((word_dic[word] + scale * emit_epsilon)) - log(sum_word_cnt)
            else:
                emit_prob[tag][word] = log((word_dic[word] + emit_epsilon)) - log(sum_word_cnt)
                
        emit_prob[tag]["UNKNOWN"] = log(scale * emit_epsilon) - log(sum_word_cnt)
    return hapax_set

def training(sentences):
    """
    Computes initial tags, emission words and transition tag-to-tag probabilities
    :param sentences:
    :return: intitial tag probs, emission words given tag probs, transition of tags to tags probs
    """
    init_prob = defaultdict(lambda: 0) # {init tag: #}
    emit_prob = defaultdict(lambda: defaultdict(lambda: 0)) # {tag: {word: # }}
    trans_prob = defaultdict(lambda: defaultdict(lambda: 0)) # {tag0:{tag1: # }}
    
    tag_list = []
    word_cnt = {}
    # TODO: (I)
    # Input the training set, output the formatted probabilities according to data statistics.
    for sentence in sentences:
        for i in range(len(sentence)):
            word, tag = sentence[i]
            # record all the tag in the set
            tag_list.append(tag)
            # * record all word in the set
            if word not in word_cnt.keys():
                word_cnt[word] = [0, tag]
            else:
                word_cnt[word][0] += 1
            if tag not in emit_prob.keys():
                emit_prob[tag] = {}
            if word not in emit_prob[tag].keys():
                emit_prob[tag][word] = 0
            emit_prob[tag][word] += 1
            if i == 0:
                if tag not in init_prob.keys():
                    init_prob[tag] = 0
                init_prob[tag] += 1 
            else:
                _, prev_state_key = sentence[i - 1]
                if prev_state_key not in trans_prob.keys():
                    trans_prob[prev_state_key] = {}
                if tag not in trans_prob[prev_state_key].keys():
                    trans_prob[prev_state_key][tag] = 0
                trans_prob[prev_state_key][tag] += 1
    unique_tag_list = list(set(tag_list))
    # probability of init 
    prob_word_tag(init_prob)
    hapax_set = smooth_emit(emit_prob, word_cnt)
    smooth_transmission(trans_prob, unique_tag_list)
    suffix_hapax_set = get_hapax_set(word_cnt, unique_tag_list)
    return init_prob, emit_prob, trans_prob, hapax_set, suffix_hapax_set

def have_profix(word):
    suffix_word_4, suffix_word_3, suffix_word_2 = word[-4:], word[-3:], word[-2:]
    if suffix_word_4 in suffix_list:
            suffix = suffix_word_4
    elif suffix_word_3 in suffix_list:
            suffix = suffix_word_3
    elif suffix_word_2 in suffix_list:
            suffix = suffix_word_2
    else:
            suffix = "UNKNOWN"
    new_type = "X_" + suffix
    return new_type

def get_hapax_set(word_cnt, tag_lst):
    hapax_set = []
    for word in word_cnt.keys():
        if word_cnt[word][0] < HAPAX_THRESH:
            hapax_set.append((word, word_cnt[word][1]))
    hapax_set = set(hapax_set)
    suffix_hapax_set = {}
    for tag in tag_lst:
        if tag == "START" or tag == "END":
            continue
        suffix_hapax_set[tag] = {}
        for suffix in suffix_list:
            new_type = "X_"+suffix
            suffix_hapax_set[tag][new_type] = 0
    for word, tag in hapax_set:
        new_type = have_profix(word)
        suffix_hapax_set[tag][new_type] += 1
    for tag in suffix_hapax_set.keys():
        num = sum(suffix_hapax_set[tag].values()) + len(suffix_list) * emit_epsilon
        for suffix in suffix_hapax_set[tag].keys():
            suffix_hapax_set[tag][suffix] = log(suffix_hapax_set[tag][suffix] + emit_epsilon) - \
                                            log(num)
    # unchanged_num = len(suffix_hapax_set.keys()) - len(suffix_list)
    return suffix_hapax_set


def viterbi_stepforward(i, word, prev_prob, prev_predict_tag_seq, emit_prob, trans_prob, hapax_set, suffix_hapax_set):
    """
    Does one step of the viterbi function
    :param i: The i'th column of the lattice/MDP (0-indexing)
    :param word: The i'th observed word
    :param prev_prob: A dictionary of tags to probs representing the max probability of getting to each tag at in the
    previous column of the lattice(path table)
    :param prev_predict_tag_seq: A dictionary representing the predicted tag sequences leading up to the previous column
    of the lattice for each tag in the previous column(trellis table)
    :param emit_prob: Emission probabilities
    :param trans_prob: Transition probabilities
    :return: Current best log probs leading to the i'th column for each tag, and the respective predicted tag sequences
    (renew the table)
    """
    log_prob = {} # This should store the log_prob for all the tags at current column (i)
    predict_tag_seq = {} # This should store the tag sequence to reach each tag at column (i)
    # state is the start/previous state of a transmission
    for cur_state_key in prev_prob.keys():
        max_prob, max_index = -100000, 'START'
        # compare different path to the same state and pick up the one
        # with the maximum prob
        for prev_state_key in prev_prob.keys():
            prev_state_prob = prev_prob[prev_state_key]
            if word not in emit_prob[cur_state_key].keys():
                known_word = "UNKNOWN"
                emit = emit_prob[cur_state_key][known_word]
                if word in hapax_set and cur_state_key != "START" and cur_state_key != "END":
                    new_type = have_profix(word)
                    emit = emit - suffix_hapax_set[cur_state_key][new_type]
            else:
                known_word = word
                emit = emit_prob[cur_state_key][known_word]
            prob = prev_state_prob + trans_prob[prev_state_key][cur_state_key] + emit_prob[cur_state_key][known_word]
            if prob > max_prob:
                max_prob = prob
                max_index = prev_state_key
        # renew the value of trellis table and path table 
        log_prob[cur_state_key] = max_prob
        if cur_state_key not in prev_predict_tag_seq.keys():
            prev_predict_tag_seq[cur_state_key] = []
        prev_predict_tag_seq[cur_state_key].append((word, max_index))
            
        # TODO: (II)
    # implement one step of trellis computation at column (i)
    # You should pay attention to the i=0 special case.
    return log_prob, prev_predict_tag_seq

def viterbi_3(train, test, get_probs=training):
    '''
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    init_prob, emit_prob, trans_prob, hapax_set, suffix_hapax_set = get_probs(train)
    
    predicts = []
    
    for sen in range(len(test)):
        predict = []
        sentence=test[sen]
        length = len(sentence)
        log_prob = {}
        predict_tag_seq = {}
        # init log prob
        for t in emit_prob:
            if t in init_prob:
                log_prob[t] = log(init_prob[t])
            else:
                log_prob[t] = log(epsilon_for_pt)
            predict_tag_seq[t] = []

        # forward steps to calculate log probs for sentence
        for i in range(length):
            log_prob, predict_tag_seq = viterbi_stepforward(i, sentence[i], log_prob, predict_tag_seq, emit_prob,trans_prob, hapax_set, suffix_hapax_set)
            
        # TODO:(III) 
        # according to the storage of probabilities and sequences, get the final prediction.
        max_tag = max(log_prob, key=log_prob.get)
        for i in range(len(predict_tag_seq["START"]) - 1, -1, -1):
            
            for tag in predict_tag_seq.keys():
                if tag == max_tag:
                    word, past_tag = predict_tag_seq[tag][i]
                    predict.append((word, tag))
                    max_tag = past_tag
                    break
        predict = predict[::-1]
        predicts.append(predict)        
        
        
    return predicts
