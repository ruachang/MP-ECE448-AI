"""
Part 1: Simple baseline that only uses word statistics to predict tags
"""
import math 

def collect_word_dic(train, word_dic, tag_dic):
    for sentence in train:
        for word, tag in sentence:
            if word not in word_dic.keys():
                word_dic[word] = {}
            if tag not in word_dic[word].keys():
                word_dic[word][tag] = 1
            else:
                word_dic[word][tag] += 1
            if tag not in tag_dic.keys():
                tag_dic[tag] = 1
            else:
                tag_dic[tag] += 1
    return word_dic, tag_dic

def prob_word_tag(word):
    sum_word_cnt = sum(word.values())
    for tag in word.keys():
        word[tag] = word[tag] / sum_word_cnt

def tag_test_word(test, word_dic, tag_dic):
    for i in range(len(test)):
        sentence = test[i]
        for j in range(len(sentence)):
            word = sentence[j]
            if word in word_dic.keys():
                tag = max(word_dic[word], key=word_dic[word].get)
            else:
                tag = max(tag_dic, key=tag_dic.get)
            sentence[j] = [word, tag]
    return test

def baseline(train, test):
    '''
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    word_dic, tag_dic = {}, {}
    collect_word_dic(train, word_dic, tag_dic)
    for word in word_dic.values():
        prob_word_tag(word)
    prob_word_tag(tag_dic)
    test = tag_test_word(test, word_dic, tag_dic)
    return test