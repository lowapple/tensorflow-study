import tensorflow as tf

x_data = [1, 2, 3]
y_data = [1, 2, 3]

W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

X = tf.placeholder(tf.float32, name="X")
Y = tf.placeholder(tf.float32, name="Y")

# X 와 Y의 상관관계
hypothesis = W * X + b

# 손실 함수
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# 경사하강법 최적화 함수를 이용해 손실값을 최소화하는 연산 그래프를 생성한다.
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train_op = optimizer.minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(100):
        _, cost_val = sess.run([train_op, cost], feed_dict={X: x_data, Y: y_data})

        print(step, cost_val, sess.run(W), sess.run(b))

    print(sess.run(hypothesis, feed_dict={X : 5}))
    print(sess.run(hypothesis, feed_dict={X : 2.5}))