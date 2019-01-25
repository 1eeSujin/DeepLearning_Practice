import tensorflow as tf
tf.set_random_seed(777)  # for reproducibility

x_data = [[73., 80., 75.],# x1    x2  x3  - 1 layer
          [93., 88., 93.],
          # x1   x2   x3 - 2 layer
          [89., 91., 90.],
          [96., 98., 100.],
          [73., 66., 70.]]
y_data = [[152.],
          # y  - 1 layer
          [185.],
          [180.],
          [196.],
          [142.]]


# placeholders for a tensor
X = tf.placeholder(tf.float32, shape=[None, 3])
# shape is 3 because there are x1,x2,x3 (각 element는 3개다)
# x의 값이 대략 n개다. => n을 tensorflow에선 None으로 표시

Y = tf.placeholder(tf.float32, shape=[None, 1])
# shape is 1, data is final exam
# n개 예측 => None

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

# Initializes global variables
sess.run(tf.global_variables_initializer())

for step in range(2001):
    cost_val, hy_val, _ = sess.run(
        [cost, hypothesis, train], feed_dict={X: x_data, Y: y_data})
    if step % 10 == 0:
        print(step, "Cost: ", cost_val, "\nPrediction:\n", hy_val)
