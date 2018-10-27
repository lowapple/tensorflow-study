import tensorflow as tf
import numpy as np

# 원-핫 코딩 사용
# [ 털, 날개 ]
x_data = np.array(
    [[0,0], [1, 0], [1, 1], [0, 0], [0, 0], [0, 1]]
)

y_data = np.array([
    [1, 0, 1], # 기타
    [0, 1, 0], # 포유류
    [0, 0, 1], # 조류
    [1, 0, 0], # 기타
    [1, 0, 0], # 기타
    [0, 0, 1]  # 조류
])

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
# W = [ 입력층(특징 수), 출력층(레이블 수) ]
W = tf.Variable(tf.random_uniform([2, 3], -1., 1.))
b = tf.Variable(tf.zeros([3]))

# L = ( X -> W ) + b
L = tf.add(tf.matmul(X, W), b)
# 평가 함수
L = tf.nn.relu(L)

# 배열 내의ㅣ 결과값들이 전체 합이 1이 되도록 만들어 준다.
model = tf.nn.softmax(L)

# 교차 엔트로피 함수를 사용
# : 예측값과 실제값 사이의 확률 분포 차이를 계산한 값
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(model), axis=1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train_op = optimizer.minimize(cost)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for step in range(100):
    sess.run(train_op, feed_dict={X: x_data, Y: y_data})

    # 손실값 출력
    if (step + 1) % 10 == 0:
        print(step + 1, sess.run(cost, feed_dict={X: x_data, Y: y_data}))
    
prediction = tf.argmax(model, axis=1)
target = tf.argmax(Y, axis=1)
print('예측값 : ', sess.run(prediction, feed_dict={X: x_data}))
print('실제값 : ', sess.run(target, feed_dict={Y: y_data}))

# 정확도
is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도 : %.2f' % sess.run(accuracy * 100, feed_dict={X: x_data, Y: y_data}))