"""
Part 4: Here should be your best version of viterbi, 
with enhancements such as dealing with suffixes/prefixes separately
"""

import math
from collections import defaultdict, Counter
from math import log
from copy import deepcopy
prefix = 3
suffix = 2
HAPAX_THRESH = 1

def data_observe(train):
    word_cnt = {}
    tag_list = {}
    prefix_dic_tmp = {}
    suffix_dic_tmp = {}
    for sentence in train:
        for i in range(len(sentence)):
            word, tag = sentence[i]
            # record all the tag in the set
            if tag not in tag_list:
                    tag_list[tag] = 1
            else:
                    tag_list[tag] += 1
            # * record all word in the set
            if word in word_cnt.keys():
                word_cnt[word] = (False, tag) 
            else:
                word_cnt[word] = (True, tag)
    hapax_set = []
    for word in word_cnt.keys():
        if word_cnt[word][0]:
            hapax_set.append((word, word_cnt[word][1]))
    for word in hapax_set:
        prefix_tag = (word[0][0: prefix], word[1])
        suffix_tag = (word[0][len(word[0]) - suffix:], word[1])
        if prefix_tag not in prefix_dic_tmp.keys():
            prefix_dic_tmp[prefix_tag] = 1
        else:
            prefix_dic_tmp[prefix_tag] += 1
        if suffix_tag not in suffix_dic_tmp.keys():
            suffix_dic_tmp[suffix_tag] = 1
        else:
            suffix_dic_tmp[suffix_tag] += 1
    prefix_dic = {}
    suffix_dic = {}
    for prefixes in prefix_dic_tmp.keys():
        if prefix_dic_tmp[prefixes] > 50:
            prefix_dic[prefixes] = prefix_dic_tmp[prefixes]
            
    for suffixes in suffix_dic_tmp.keys():
        if suffix_dic_tmp[suffixes] > 50:
            suffix_dic[suffixes] = suffix_dic_tmp[suffixes]
    for key in prefix_dic.keys():
        print(key, prefix_dic[key], prefix_dic[key] / tag_list[key[1]])
    print("==================================")
    print("==================================")
    print("==================================")
    print("==================================")
    print("==================================")
    for key in suffix_dic.keys():
        print(key, suffix_dic[key], suffix_dic[key] / tag_list[key[1]])
        # print(word, word[0][0: prefix], word[0][len(word[0]) - suffix:], tag)

# Note: remember to use these two elements when you find a probability is 0 in the training data.
epsilon_for_pt = 1e-5
emit_epsilon = 1e-5  # exact setting seems to have little or no effect

# * chosen prefix and suffix list
# suffix_list = ["ly", "es", "ous", "ed", "ness", "'s", "us", "ing", "ment", "able", "er", "ic", "est", "less", "ist", "ent"]
# suffix_list = set(["ly", "es", "ous", "ed", "'s", "us", "ing", "able", "er", "ic", "est", "less", "ent"])
suffix_list = set(["ment", "ly", "ous", "able", "less", "ness", "est", "ist", "'s", "ing", "ed"])
# suffix_list = set([])

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

def get_hapax_set(word_cnt):
    hapax_set = []
    for word in word_cnt.keys():
        if word_cnt[word][0] == 1:
            hapax_set.append(word)
    hapax_set = set(hapax_set)
    suffix_hapax_set = {}
    for word in hapax_set:
        # TEST
        new_type = have_profix(word)
        if new_type not in suffix_hapax_set.keys():
            suffix_hapax_set[new_type] = []
        suffix_hapax_set[new_type].append(word)
    # unchanged_num = len(suffix_hapax_set.keys()) - len(suffix_list)
    return hapax_set, suffix_hapax_set


def smooth_emit(emit_prob, word_cnt):
    increment = HAPAX_THRESH

    hapax_set, suffix_hapax_set = get_hapax_set(word_cnt)
    suffix_tag_set = {}
    for tag in emit_prob.keys():
        suffix_tag_set[tag] = {"X_UNKNOWN": 1}
    for word in word_cnt.keys():
        if word in hapax_set:
            tag = word_cnt[word][1]
            new_type = have_profix(word)
            if new_type not in suffix_tag_set[tag].keys():
                suffix_tag_set[tag][new_type] = 0 
    # TEST
            suffix_tag_set[tag][new_type] += increment
    hapax_scale_set = {}
    for tag in emit_prob.keys():  
        sum_num = sum(emit_prob[tag].values())
        for word in emit_prob[tag].keys():
            if word in hapax_set:
                new_type = have_profix(word)
                # scale = 1
                # scale = suffix_tag_set[tag][new_type] / len(suffix_hapax_set[new_type])
                sum_word_cnt = sum_num + emit_epsilon * (len(emit_prob[tag].keys()) + 1 )
                
                # scale = emit_prob[tag][new_type] / (len(hapax_set) - unknown_num)
                emit_prob[tag][word] = log((1 + emit_epsilon)) - log(sum_word_cnt) #+ log()
            else:
                sum_word_cnt = sum_num + emit_epsilon * (len(emit_prob[tag].keys()) + 1 )
                emit_prob[tag][word] = log((emit_prob[tag][word] + emit_epsilon)) - log(sum_word_cnt)
        scale = suffix_tag_set[tag]["X_UNKNOWN"] / len(suffix_hapax_set["X_UNKNOWN"])
        sum_word_cnt = sum_num + scale * emit_epsilon * (len(emit_prob[tag].keys()) + 1 )
        
        emit_prob[tag]["UNKNOWN"] = log( scale * emit_epsilon) - log(sum_word_cnt) #+ log(scale)
        # TEST
        # for word in emit_prob[tag].keys():
        #     suffix_word_4, suffix_word_3, suffix_word_2 = word[-4:], word[-3:], word[-2:]
        #     if "X_" in word:
        #         new_type = word
        #         scale = emit_prob[tag][new_type] / len(suffix_hapax_set[new_type])
        #         emit_prob[tag][new_type] = ((emit_prob[tag][new_type] + scale * emit_epsilon)) / (sum_word_cnt)
        #     elif word in hapax_set:
        #         if suffix_word_4 in suffix_list:
        #             new_type = "X_" + suffix_word_4
        #             scale = emit_prob[tag][new_type] / len(suffix_hapax_set[new_type])
        #         elif suffix_word_3 in suffix_list:
        #             new_type = "X_" + suffix_word_3
        #             scale = emit_prob[tag][new_type] / len(suffix_hapax_set[new_type])
        #         elif suffix_word_2 in suffix_list:
        #             new_type = "X_" + suffix_word_2
        #             scale = emit_prob[tag][new_type] / len(suffix_hapax_set[new_type])
        #         else:          
        #             scale = minor_num / unknown_num
        #             emit_prob[tag][word] = ((emit_prob[tag][word] + scale * emit_epsilon)) / (sum_word_cnt)
        #     else:
        #         emit_prob[tag][word] = ((emit_prob[tag][word] + emit_epsilon)) / (sum_word_cnt)
        # scale = minor_num / unknown_num
        # emit_prob[tag]["UNKNOWN"] = (scale * emit_epsilon) / (sum_word_cnt)
        # prob_sum = sum(emit_prob[tag].values())
    return suffix_hapax_set, suffix_tag_set

def update_emission_probabilities_viterbi_3(emit_prob, word_cnt, emit_epsilon):
    hapax_set = get_hapax_set(word_cnt)
    for tag, word_dic in emit_prob.items():
        num_minor = max(sum(1 for word in word_dic if word in hapax_set), 1)
        num_major = len(word_dic) - num_minor
        scale = num_minor / len(hapax_set)
        
        sum_word_cnt = sum(word_dic.values()) + emit_epsilon * (len(word_dic) + 1)
        
        for word, count in word_dic.items():
            if word in hapax_set:
                # Check for patterns or suffixes and map to pseudowords
                if word.endswith("-ing"):
                    pseudoword = "X-ING"
                    # Adjust the increment based on the pseudoword
                    word_prob = log(count + 0.5 * emit_epsilon) - log(sum_word_cnt)
                # Add more pattern checks here as needed
                
                else:
                    pseudoword = "UNKNOWN"
                    word_prob = log(count + scale * emit_epsilon) - log(sum_word_cnt)
                
                # Update the emission probability for the pseudoword
                emit_prob[tag].setdefault(pseudoword, 0)
                emit_prob[tag][pseudoword] += word_prob
            else:
                # Handle known words with the existing logic
                emit_prob[tag].setdefault(word, 0)
                emit_prob[tag][word] = log(count + emit_epsilon) - log(sum_word_cnt)

        # Handle the "UNKNOWN" pseudoword for unseen words
        emit_prob[tag].setdefault("UNKNOWN", 0)
        emit_prob[tag]["UNKNOWN"] = log(scale * emit_epsilon) - log(sum_word_cnt)

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
            if word in word_cnt.keys():
                word_cnt[word] = (word_cnt[word][0] + 1, tag)
            else:
                word_cnt[word] = (1, tag)
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
    suffix_hapax_set, suffix_tag_set = smooth_emit(emit_prob, word_cnt)
    smooth_transmission(trans_prob, unique_tag_list)
    # update_emission_probabilities_viterbi_3(emit_prob, word_cnt, emit_epsilon)
    return init_prob, emit_prob, trans_prob, suffix_hapax_set, suffix_tag_set

def viterbi_stepforward(i, word, prev_prob, prev_predict_tag_seq, emit_prob, trans_prob, suffix_hapax_set, suffix_tag_set):
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
            if word in emit_prob[cur_state_key].keys():
                known_word = word
                bias = 0
            else:
                new_type = have_profix(word)
                if new_type in suffix_tag_set[cur_state_key].keys():
                    scale = suffix_tag_set[cur_state_key][new_type] / len(suffix_hapax_set[new_type])
                else:
                    scale = emit_epsilon / len(suffix_hapax_set[new_type])
                bias = log(scale)
                
                known_word = "UNKNOWN" 
            prob = prev_state_prob + trans_prob[prev_state_key][cur_state_key] + emit_prob[cur_state_key][known_word] + bias
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
    init_prob, emit_prob, trans_prob, suffix_hapax_set, suffix_tag_set = get_probs(train)
    
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
            log_prob, predict_tag_seq = viterbi_stepforward(i, sentence[i], log_prob, predict_tag_seq, emit_prob, trans_prob, suffix_hapax_set, suffix_tag_set)
            
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