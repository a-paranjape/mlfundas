# Fundamentals of Machine Learning

Modules implementing basic machine learning algorithms for classification and regression, including Perceptron and standard feed-forward neural network (NN). Training of NNs uses error back-propagation with stochastic gradient descent and optional Adam optimization.

## Code organization
* `mllib`: basic utilities for array manipulation and algorithm evaluation.
* `mlmodules`: building-block modules (activation, loss, regularization, linear).
* `mlalgos`: algorithms
    * `Perceptron` -- classic linear classification.
    * `Sequential` -- fully-connected feed-forward NN (subsets of this are logistic regression, support vector machine, ridge regression).
    * `BuildNN` -- wrapper to systematically search over various classes of NN architecture.
    * `BiSequential` -- architecture to approximate functions $f(x,\theta)$ using a separable basis of the form $f(x,\theta) \approx \sum_{i=1}^{n} b_i(x) w_i(\theta)$ 
* `mlseq`: sequential algorithms (state machines, recurrent NN, Markov decision process, ...). **UNDER CONSTRUCTION**
* `utilities`: provides various utility modules for i/o and simple tools.

This code arose from following the excellent course on [Introduction to Machine Learning](https://openlearninglibrary.mit.edu/courses/course-v1:MITx+6.036+1T2019/course/) by L. Kaebling and colleagues, hosted by the [MIT Open Learning Library](https://openlearninglibrary.mit.edu/).

## Examples
* $\texttt{examples/BiSequential.ipynb}$: Jupyter notebook showing example usage of existing `BiSequential` instance, based on [Paranjape & Sheth (2024)](https://arxiv.org/abs/2410.21374).

## Contact
Aseem Paranjape: aseem_at_iucaa_dot_in
