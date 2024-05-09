# Fundamentals of Machine Learning

Modules implementing basic machine learning algorithms for classification and regression, including `Perceptron`, logistic regression, support vector machines, ridge regression and standard feed-forward neural network (NN) with error back-propagation using `stochastic gradient descent`.

## Code organization
* `mllib`: basic utilities for array manipulation and algorithm evaluation.
* `mlmodules`: building-block modules (activation, loss, regularization, linear).
* `mlalgos`: algorithms -- `Perceptron`, fully-connected feed-forward NN (subsets of this are logistic regression, support vector machine, ridge regression), wrapper to systematically search over (restricted) class of NNs.
* `mlseq`: sequential algorithms (state machines, recurrent NN, Markov decision process, ...). **UNDER CONSTRUCTION**
* `utilities`: provides various utility modules for i/o and simple tools.

This code arose from following the excellent course on [Introduction to Machine Learning](https://openlearninglibrary.mit.edu/courses/course-v1:MITx+6.036+1T2019/course/) by L. Kaebling and colleagues, hosted by the [MIT Open Learning Library](https://openlearninglibrary.mit.edu/).

## Contact
Aseem Paranjape: aseem_at_iucaa_dot_in
