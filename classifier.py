import tensorflow as tf
import numpy as np
def sigmoid(x):
    return(1/(1+tf.pow(np.e,x)))

def learn(trainingSet):

    # Parameters
    learning_rate = 0.01
    training_epochs = 25
    batch_size = 100
    display_step = 1

    # tf Graph Input
    #x = tf.placeholder( [None, 7])  # mnist data image of shape 28*28=784
    #y = tf.placeholder( [None, 1])

    # Set model weights
    W = tf.Variable(tf.zeros([7, 1]))
    b = tf.Variable(tf.zeros([1]))

    concatMatrix=tf.concat(W, b, 1)

    pred = tf.nn.softmax(tf.matmul(trainingSet,concatMatrix ))  # Softmax
    h= sigmoid(pred)


