import tensorflow as tf
import numpy as np
from tensorflow.python import debug as tf_debug

n=0
m=0

def frame_into_batch(frame):
    training_array= frame.values.transpose()
    X_values = training_array[1:, :]
    y_values = training_array[0, :]
    return(X_values,y_values)

"""expects data frame"""
def learn(training_frame,test_frame):

    # Parameters
    learning_rate = 0.02
    n = len(training_frame.keys()) - 1


    X = tf.placeholder(tf.float32, (n, None))
    m = tf.shape(X)[1]
    X_normalized = tf.nn.l2_normalize(X,1)

    y = tf.placeholder(tf.float32, (None,))

    W = tf.Variable(tf.zeros([1, n]))
    b = tf.Variable(tf.zeros([1,1]))


    X_with_bias = tf.concat([tf.ones([1, m]),X_normalized], 0)
    Theta=tf.concat([b,W], 1)

    h_of_x= tf.sigmoid(tf.einsum('ij,ki->j', X_with_bias, Theta))
    
    cost= 1/m * tf.cast(tf.reduce_sum(
        tf.einsum("i,i->",-1*tf.transpose(y), tf.log(h_of_x))
        - tf.einsum("i,i->",1-tf.transpose(y), tf.log(1 - h_of_x))), tf.float64)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    init = tf.global_variables_initializer()

    # Start training
    with tf.Session() as sess:
        # Run the initializer


        sess.run(init)
        #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        X_values,y_values= frame_into_batch(training_frame)
        c = sess.run(cost, feed_dict={X: X_values,
                                      y: y_values})
        print("training set J: ", c)
        _=sess.run(optimizer,feed_dict={X: X_values,
                                        y: y_values})


        c = sess.run(cost, feed_dict={X: X_values,
                                      y: y_values})
        print("training set J: ", c)

        #i=0
        # while(c[i]!= 1.0):
        #     i=i+1
        # print("fucked up sigmoid: ",i) #258




