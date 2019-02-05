import tensorflow as tf
import numpy as np


x_data = [[1,2,1,1],[2,1,3,2],[3,1,3,4],[4,1,5,5],[1,7,5,5],[1,2,5,6],[1,6,6,6],[1,7,7,7]]
y_data = [[0,0,1],[0,0,1],[0,0,1],[0,1,0],[0,1,0],[1,0,0],[1,0,0]]
# y data 는 0,1을 주어줬지만, 여기엔 여러개의 값들이 있음
# one Hot을 사용 (one Hot : 3자리를 만들고, 하나만 핫하게 한다.)
# 1인경우 => 두번쨰자리맛 핫하게해줌
# 2인경우 => 3번째자리만 핫하게해준다
# one hot encoding : 하나의 자리만 핫하게 해줌


X = tf.placeholder("float", shape=[None,4])
Y = tf.placeholder("float", shape = [None,3])
nb_classes = 3

W = tf.Variable(tf.random_normal([4,nb_classes]),name='weight')
b = tf.Variable(tf.random_normal([nb_classes]),name='bias')

# tf.nn.softmax comptes softmax activations
# softmax = exp(logits) / reduce_sum(exp(Logits),dim)

hypothesis = tf.nn.softmax(tf.matumal(X,W)+b)
tf.matumul(X,W)+b


# Cross entropy cost/loss
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis),axis=1))



optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        sess.run(optimizer,feed_dict={X:x_data,Y:y_data})
        if step % 200 == 0:
            print(step,sess.run(cost,feed_dict={X:x_data,Y:y_data}))




