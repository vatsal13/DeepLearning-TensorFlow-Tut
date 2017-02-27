# Using MNIST data-set 28x28
import tensorflow as tf

'''
Feed Forward
input data > weight > hidden layer 1 (activation function) > weights > hidden layer 2/
(actiavtion function) > weights > output layer

> compare output with intended output using cost function (cross entropy)

>optimization to minimize cost (Adam, SGD, AdaGrad)
(Backpropagation)
'''

from tf.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data", one_hot=True)

'''
one hot vector used for output

0 = [1,0,0,0,0,0,0,0,0,0]
1 = [0,1,0,0,0,0,0,0,0,0]
2 = [0,0,1,0,0,0,0,0,0,0]
..
..
9 = [0,0,0,0,0,0,0,0,0,1]
'''


