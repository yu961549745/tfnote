""" TensorFlow 模型导入导出 """

#%% 定义模型
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#%% 训练和保存模型
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for _ in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    print(sess.run(accuracy, feed_dict={
        x: mnist.test.images, y_: mnist.test.labels}))
    saver = tf.train.Saver()
    saver.save(sess, 'model/my-model')
    saver.export_meta_graph('model/my-model.meta')

#%% 加载和运行模型
sess = tf.Session()
new_saver = tf.train.import_meta_graph('model/my-model.meta')
new_saver.restore(sess, 'model/my-model')
print(sess.run(accuracy, feed_dict={
    x: mnist.test.images, y_: mnist.test.labels}))
