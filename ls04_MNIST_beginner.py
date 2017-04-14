""" MNIST for beginners """

#%% 下载和读取数据
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#%% 定义模型
import tensorflow as tf

# 定义输入变量，None表示任意维度，每个图片是一个784维的行
x = tf.placeholder(tf.float32, [None, 784])

# 定义模型参数
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# 定义输出变量
y = tf.matmul(x, W) + b

#%% 训练模型
# 答案
y_ = tf.placeholder(tf.float32, [None, 10])
# 采用交叉熵作为损失函数
# 不直接使用下面的公式是因为它数值不稳定
# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ *tf.log(y), reduction_indices=[1]))
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

# tensorflow 使用反向传播算法来训练网络

# 使用梯度下降算法来优化损失函数
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# 启动交互式会话
sess = tf.InteractiveSession()

# 初始化变量
tf.global_variables_initializer().run()

# 训练
for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

#%% 评估模型
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={
      x: mnist.test.images, y_: mnist.test.labels}))
