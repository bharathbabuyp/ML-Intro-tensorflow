# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 22:22:51 2017

@author: Lenovo
"""

import tensorflow as tf
import numpy as np

##creating a dataset
x_data = np.random.rand(5000).astype(np.float32)
y_data = 13*(x_data**2) + 137*x_data + 9


## initializing random values to the variables
theta1=tf.Variable(tf.random_uniform([1],0,30))
theta2=tf.Variable(tf.random_uniform([1],0,200))
theta3=tf.Variable(tf.random_uniform([1],0,10))

y= theta1*(x_data**2)+theta2*(x_data)+theta3   ###hypothesis for prediction


#mean of the cost for all examples
loss=tf.reduce_mean(tf.square(y-y_data))  ## loss or cost


# initialzing an object for optimizing the loss or cost function
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init=tf.global_variables_initializer()

sess = tf.Session()

sess.run(init)


iteration=2000

for step in range(iteration):
    sess.run(train)
    if step%10==0:
        print(step, sess.run(theta1), sess.run(theta2), sess.run(theta3))
    
## done

#done 2