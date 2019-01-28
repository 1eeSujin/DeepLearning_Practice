import tensorflow as tf
tf.set_random_seed(777)  # for reproducibility

# 파일 하나를 넣는 곳
filename_queue = tf.train.string_input_producer(
    ['data-01-test-score.csv'], shuffle=False, name='filename_queue')

# reader를 정의 하는 부분
reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

# Default values, in case of empty columns. Also specifies the type of the
# decoded result.
# 각각 필드에 데이터 타입
record_defaults = [[0.], [0.], [0.], [0.]]
xy = tf.decode_csv(value, record_defaults=record_defaults)

# collect batches of csv in
train_x_batch, train_y_batch = \
# batch를 가지고 데이터를 읽음
    tf.train.batch([xy[0:-1], xy[-1:]], batch_size=10)
    #               x 데이타     y데이타 ,  한번에 몇개씩 가져올지
    # 한번 펌프할때마다 몇개씩 가져올지 => batch_size

# placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, 3])
# x value가 몇개인지 꼭 맞춰주기

Y = tf.placeholder(tf.float32, shape=[None, 1])
# y value가 몇개인지 꼭 맞춰주기


W = tf.Variable(tf.random_normal([3, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Hypothesis
hypothesis = tf.matmul(X, W) + b

# Simplified cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

# Launch the graph in a session.
sess = tf.Session()
# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())

# queue를 관리하는 부분  (일반적으로 쓰는 부분)
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

# session을 가지고 실행 시킴
# 이 값을 추후에 feed_dict을 이용해서 값을 넣어줌

for step in range(2001):
    # 펌프를 해서, 데이터를 가져온다.
    x_batch, y_batch = sess.run([train_x_batch, train_y_batch])
    cost_val, hy_val, _ = sess.run(
        [cost, hypothesis, train], feed_dict={X: x_batch, Y: y_batch})
    if step % 10 == 0:
        print(step, "Cost: ", cost_val, "\nPrediction:\n", hy_val)

coord.request_stop()
coord.join(threads)

# Ask my score
print("Your score will be ",
      sess.run(hypothesis, feed_dict={X: [[100, 70, 101]]}))

print("Other scores will be ",
      sess.run(hypothesis, feed_dict={X: [[60, 70, 110], [90, 100, 80]]}))

'''
Your score will be  [[185.33531]]
Other scores will be  [[178.36246]
 [177.03687]]
'''