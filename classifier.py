import tensorflow as tf
import numpy as np

def learn(trainingSet):

    # Parameters
    learning_rate = 0.01
    training_epochs = 25
    batch_size = 100
    display_step = 1

    # tf Graph Input
    #x = tf.placeholder( [None, 7])  # mnist data image of shape 28*28=784
    #y = tf.placeholder( [None, 1])



    n= len(trainingSet.keys())
    m= len(trainingSet[trainingSet.keys()[0]])

    # Set model weights
    trainngSetWithBias= tf.concat([tf.ones([1,trainingSet]),trainingSet],1)
    W = tf.Variable(tf.zeros([n, 1]))
    b = tf.Variable(tf.zeros([1,1]))

    Theta=tf.concat([W, b], 0)

    pred = tf.nn.softmax(tf.matmul(trainngSetWithBias,Theta ))  # Softmax
    h= tf.sigmoid(pred)


