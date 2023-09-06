# Bigram Mixture Model

Unigram of Naive Bayes ignores the sequence of the word and only pays attention to the frequency of the words. To take the order into consideration, we use bigram mixture model.

Bigram model uses the word pair to make classification. For the same dataset, the training data squres so the accuracy may improve significantly.

To use bigram pridiction model, how to calculte the probobility keeps the same with the unigram. So just add the probobility of the word pairs is ok. 

To mix the unigram and the bigram, the program will use the parameter $\lambda$ to adjust the contribution of unigram and bigram. 

Finally, the probability becomes

$P(y|word) = (1 - \lambda) * log(P(y | word)) + \lambda * log(P(y | word pairs))$