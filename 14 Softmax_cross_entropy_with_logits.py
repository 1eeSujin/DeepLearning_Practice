import tensorflow as tf
import numpy as np

# Predicting animal type based on various features
xy = np.loadtxt('data-04-zoo.csv',delimiter=',',dtype=np.float32)
x_data = xy[:,0:-1]
y_data = xy[:,[-1]]

# 0~ 6까지 숫자를 가지니까 class가 7
nb_classes = 7

# x가 16개
X = tf.placeholder(tf.int32,[None,16])
# y를 one hot으로 바꿈
Y = tf.paceholder(tf.int32,[None,1])

# one hot으로 바굼
Y_one_hot = tf.one_hot(Y,nb_classes)
# shape을 우리가 원하는대로 바꿔줌
Y_one_hot = tf.reshape(Y_one_hot,[-1,nb_classes])

# 입력 16개, 출력은 7개 weight이 됨
# shape을 한 번 더 해줌
W=tf.Variable(tf.random_normal([16,nb_classes]),name='weight')
# 7
b = tf.Variable(tf.random_normal([nb_classes]),name='bias')

# tf.nn.softmax computes softmax activations
logits = tf.matmul(X,W)+b
hypothesis = tf.nn.softmax(logits)

# cross entropy cost/loss
cost_i = tf.nn.softmax_cross_entropy_with_logits(logits =logits,labels=Y_one_hot)
# 최종 cost
cost = tf.reduce_mean(cost_i)
# 최소화
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)


# -----------학 습 ---------------

# 예측한 값이 맞는 지 확인하고 싶음
# probability를 0~6에 있는 값
prediction = tf.argmax(hypothesis,1)
# Y_one_hot중에 하나를 선택한 값 과 prediction을 비교
# Y one hot = label , 1 : Y
correct_prediction = tf.equal(prediction,tf.argmax(Y_one_hot,1))
# 값을 비교해서 평균을 냄 : accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

# Launch graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2000):
        # optimizer를 실행시킬 때, x데이터 y데이터를 한번씩 실행시키게 됨
        sess.run(optimizer,feed_dict={X:x_data,Y:y_data})
        if step % 100 == 0:
            # 100씩마다 출력
            loss,axx = sess.run([cost,accuracy],feed_dict={
                X:x_data,Y:y_data
            })
            print("Step : {:5\tLoss:{:.3f}\tAcc:{:.2%}".format(step,loss,acc))

# ---------------------
    # Let's see if we can predict (우리가 잘 예측 햇는지 화인)
    pred = sess.run(prediction,feed_dict={X:x_data})
    #y_data : (N,1) = flatten => (N, ) matches pred.shape
    # y가  [[],[]] => [  ] 이 형태로 y를 바꾸게 해줌 : flatten
    # zip으로 각각의 리스트를 묶어서, 각각의 엘리먼트를 p로 보내기 위해 zip을 함
    for p,y in zip(pred, y_data.flatten()):
        print("[{}] Prediction: {} True Y : ()".format(p==int(y),p,int(y)))



