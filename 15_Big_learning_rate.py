import tensorflow as tf
import numpy as np

x_data = [[1,2,1],[1,3,2],[1,3,4,],[1,5,6],[1,7,5],[1,2,5],[1,6,6],[1,7,7]]
y_data = [[0,0,1],[0,0,1],[0,0,1],[0,1,0],[0,1,0],[0,1,0],[1,0,0],[1,0,0]]

x_test = [[2,1,1],[3,1,2],[3,3,4]]
y_test = [[0,0,1],[0,0,1],[0,0,1]]

X = tf.placeholder("float",[None,3])
Y = tf.placeholder("float",[None,3])



W = tf.Variable(tf.random_normal([3,3]))
b = tf.Variable(tf.random_normal([3]))

hypothesis = tf.nn.softmax(tf.matmul(X,W)+b)
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis),axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=10.0).minimize(cost)
# learning rate를 아주 크게 줌

#  Correct prediction Test model
prediction = tf.arg_max(hypothesis,1)
# 예측한 것이 맞는지 안맞는지 확인
is_correct = tf.equal(prediction,tf.arg_max(Y,1))
accuracy = tf.reduce_mean(tf.cast(is_correct,tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(201):
        # x_data, y_data = training data 임
        cost_val,W_val,_ = sess.run([cost,W,optimizer],feed_dict={X:x_data,Y:y_data})
        print(step,cost_val,W_val)

    # 여기서 나오는 Accuracy는 test data를 던져줫을 때 나오는 predict
    print("Prediction :",sess.run(prediction,feed_dict={X:x_test}))
    print("Accuracy: ",sess.run(accuracy,feed_dict={X:x_test,Y:y_test}))






