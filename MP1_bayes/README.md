# MP1: Use Naive Bayes to predict the positive reviews

## Background

Using the bag of words
* the text => bag of ***independent*** words
* only ***frequency***, ignore the position
* $P(Type = k|x_1, x_2...x_n) = P(Type = k) \times \prod\limits_{i \ from \ 0 \ to \ n - 1}P((x_i) | P(Type = k))$

  * posterior prob = likeliness * prob

## Theoritical analyze

During the training stages, we should learn
* the likeliness of each word. Like when the tyte is positive, what words will occure and their prob($P(x_i | P(positive))$)
* the prob of kinds is set as input param

After gaining the likeliness of prob of each $x_i$, we can try to predict in development phase and make some adjustment to the parameters. We need to calculate
* P(kinds | x_1, x_2, x_n) and its corresponding prob

## Some programming detail

* Use some data structure like counter
* Use `log` to calculate the times of the prob => turn it into add
* To avoid zero prob in prediction, we have to use ***Laplace smoothing*** to smoth the prob 
  * Assume that each kinds at least have $\alpha$ items. Then the sum of the samples has changed to $\ n + \alpha * (k)n\ $So the likeliness becomes $$