import tensorflow as tf
import numpy as np

"""expects data frame"""
def learn(training_frame):

    # Parameters
    learning_rate = 0.01
    training_epochs = 25
    batch_size = 100
    display_step = 1

    # tf Graph Input
    #x = tf.placeholder( [None, 7])  # mnist data image of shape 28*28=784
    #y = tf.placeholder( [None, 1])

    training_array= training_frame.values.transpose()
    n = training_array.shape[0]-1
    m = training_array.shape[1]

    #y=training_array[0,:].astype(np.float32)
    y=tf.placeholder(tf.float32,(m))
    X = tf.placeholder(tf.float32, (1+n, m))

    W = tf.Variable(tf.zeros([1, n]))
    b = tf.Variable(tf.zeros([1,1]))

    # add bias
    #X = tf.concat([tf.ones([1, m]), X], 0)
    Theta=tf.concat([b,W], 1)
    h_of_x= tf.einsum('ij,ki->j', X,Theta )

    pred = tf.nn.sigmoid(h_of_x)  # Softmax

    #J = 1 / m * sum( -1 * y' * log(h_of_x) - (1-y') * log(1 - h_of_x) );
    J= 1/m * tf.reduce_sum(
        tf.einsum("i,i->",-1*tf.transpose(y), tf.log(h_of_x))
        - tf.einsum("i,i->",1-tf.transpose(y), tf.log(1- h_of_x)) )


