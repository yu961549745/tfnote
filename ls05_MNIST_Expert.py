""" 卷积神经网络训练MNIST """
#%%
# 准备工作
# 定义常量
import os
import shutil
import time
dataPath = 'MNIST_data'
modelSavePath = 'MNIST_conv'
modelCkpPath = os.path.join(modelSavePath, 'conv')
modelMetaFile = modelCkpPath + ".meta"
trainSteps = 500

# 读取数据
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(dataPath, one_hot=True)

# 启动会话
# InteractiveSession更适合在交互式环境下使用
import tensorflow as tf

# 定义CNN


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(input_image, W):
    return tf.nn.conv2d(input_image, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(input_image):
    return tf.nn.max_pool(input_image, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


input_image = tf.placeholder(tf.float32, [None, 784], name='input_image')
output_valid = tf.placeholder(tf.float32, [None, 10], name='valid_output')

with tf.name_scope('hidden1'):
    # 第一卷积层
    # 5,5 表示卷积的面片大小，1表示输入channels，32表示输出 channels
    W_conv1 = weight_variable([5, 5, 1, 32])
    # 对应的偏置向量
    b_conv1 = bias_variable([32])
    #  将input_image变成一个4维张量
    x_image = tf.reshape(input_image, [-1, 28, 28, 1])
    # 将 x_image 和 W_conv1 进行卷积，经过max_pool，图像大小变成14*14
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

with tf.name_scope('hidden2'):
    # 第二层卷积，提取64个特征，图像大小变成7*7
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

with tf.name_scope('full_link'):
    # 全连接层，将64个特征，用1024个神经元，套上ReLU进行输出
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    # dropout 防止过拟合
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

with tf.name_scope('linear_output'):
    # 输出层
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    output = tf.add(tf.matmul(h_fc1_drop, W_fc2), b_fc2, name='output')

with tf.name_scope('train'):
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=output_valid, logits=output))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

with tf.name_scope('valid'):
    correct_prediction = tf.equal(
        tf.argmax(output, 1), tf.argmax(output_valid, 1))
    accuracy = tf.reduce_mean(
        tf.cast(correct_prediction, tf.float32), name='accuracy')

# 模型训练
if os.path.exists(modelSavePath):
    shutil.rmtree(modelSavePath)
os.mkdir(modelSavePath)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

# 添加 summary
tf.summary.scalar('loss', cross_entropy)
summary = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter(modelSavePath, sess.graph)
saver = tf.train.Saver()

# 训练
st = time.time()
for step in range(trainSteps):
    batch = mnist.train.next_batch(50)
    if step % 100 == 0 or step == trainSteps - 1:
        _, loss_value, train_accuracy, summary_str = sess.run([train_step, cross_entropy, accuracy, summary], feed_dict={
            input_image: batch[0], output_valid: batch[1], keep_prob: 0.5})
        print("step = %d, loss = %g, train_accuracy = %g, time=%.3f sec" %
              (step, loss_value, train_accuracy, time.time() - st))
        summary_writer.add_summary(summary_str, step)
        summary_writer.flush()
    else:
        sess.run(train_step, feed_dict={
            input_image: batch[0], output_valid: batch[1], keep_prob: 0.5})

print("test accuracy %g" % accuracy.eval(feed_dict={
    input_image: mnist.test.images, output_valid: mnist.test.labels, keep_prob: 1.0}))

# 保存模型
saver.save(sess, modelCkpPath)
saver.export_meta_graph(modelMetaFile)
sess.close()
