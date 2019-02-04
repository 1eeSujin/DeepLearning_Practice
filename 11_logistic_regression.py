import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x_data = [[1,2],[2,3],[3,1],[4,3],[5,3],[6,2]]
y_data = [[0],[0],[0],[1],[1],[1]]

# placeholders for a tensor that will be always fed.
x = tf.placeholder(tf.float32,shape=[None,2])
y = tf.placeholder(tf.float32,shape=[None,1])

# x에 2개의 값이 들어오니까, y가 1개니까 [2,1]의 weight을 설정
W = tf.Variable(tf.random_normal([2,1]),name='weight')
# bias는 항상 나가는 값과 같음
b = tf.Variable(tf.random_normal([1]),name='bias')

# Hypothesis using sigmoid
# tf.div(1.,1. + tf.exp(tf.matmul(X,W)+b))와 같은 수식
hypothesis = tf.sigmoid(tf.matmul(x , W)+b)

# hypothesis을 사용하여 cost/loss function
# tf.reduce_mean : 평균
# Y * tf.log  : 로그부분
cost = -tf.reduce_mean(y * tf.log(hypothesis)+(1-y) * tf.log(1-hypothesis))

# GradienDescentOptimizer : 미분
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

predicted = tf.cast(hypothesis>0.5,dtype=tf.float32)

# 예측한 값들이 얼마나 정확한지 알아보기 위해
# 예측한 값과 Y가 얼마나 똑같은지 => TRUE FALSE 값
# 전체 평균내면
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted,y),dtype=tf.float32))

# Launch graph
with tf.Session() as sess:
    #Variable 초기화
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        # cost 값을 variable로 저장
        cost_val,_ = sess.run([cost,train],feed_dict={x:x_data,y:y_data})
        # 200번 마다 한번씩 출력
        if step % 200 == 0:
            print(step, cost_val)

    # 학습된 모델을 가지고 (x,y값을 가지고) hypothesis 값, 예측한 값과 y를 비교해서 accuracy가 뭐가 나올지 출력
    h,c,a = sess.run([hypothesis,predicted,accuracy],feed_dict={x:x_data,y:y_data})
    print("\nHypothesis :",h,"\nCorrect (Y):",c,"\nAccuracy:",a)


# 출력 값이 실제 y값과 비교하면 동일