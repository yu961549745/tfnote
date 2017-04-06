"""
训练和保存模型
"""
#%% 训练和保存模型
# 定义常量
import os
import shutil
dataPath = 'MNIST_data'
modelDir = 'MNIST_softmax_model'
modelName = 'softmax'
modelPath = os.path.join(modelDir, modelName)
modelFile = modelPath + ".meta"


# 加载数据
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(dataPath, one_hot=True)

# 定义模型
import tensorflow as tf
with tf.name_scope('linear_softmax'):
    input = tf.placeholder(tf.float32, [None, 784], name='input')
    W = tf.Variable(tf.zeros([784, 10]), name="weights")
    b = tf.Variable(tf.zeros([10]), name='biases')
    output = tf.nn.softmax(tf.matmul(input, W) + b, name='output')
with tf.name_scope('train'):
    valid = tf.placeholder(tf.float32, [None, 10], name='valid')
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=valid, logits=output))
    train_step = tf.train.GradientDescentOptimizer(
        0.5, name="gd_train").minimize(cross_entropy)
with tf.name_scope('valid'):
    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(valid, 1))
    accuracy = tf.reduce_mean(
        tf.cast(correct_prediction, tf.float32), name='accuracy')

# 训练和保存模型
if os.path.exists(modelDir):
    shutil.rmtree(modelDir)
os.mkdir(modelDir)
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for _ in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={input: batch_xs, valid: batch_ys})
    print(sess.run(accuracy, feed_dict={
        input: mnist.test.images, valid: mnist.test.labels}))
    saver = tf.train.Saver()
    saver.save(sess, modelPath)
    saver.export_meta_graph(modelFile)
    tf.summary.FileWriter(modelDir, sess.graph)
