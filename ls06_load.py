'''
加载模型，使用和继续训练
'''
# 相关常量
import os
dataPath = 'MNIST_data'
modelDir = 'MNIST_softmax_model'
modelName = 'softmax'
modelPath = os.path.join(modelDir, modelName)
modelFile = modelPath + ".meta"

# 加载数据
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(dataPath, one_hot=True)

# 加载模型
import tensorflow as tf
sess = tf.Session()
new_saver = tf.train.import_meta_graph(modelFile)
new_saver.restore(sess, modelPath)

# 提取相关变量
input = sess.graph.get_tensor_by_name('linear_softmax/input:0')
output = sess.graph.get_tensor_by_name('linear_softmax/output:0')
valid = sess.graph.get_tensor_by_name('train/valid:0')
accuracy = sess.graph.get_tensor_by_name('valid/accuracy:0')

print(sess.run(accuracy, feed_dict={
    input: mnist.test.images, valid: mnist.test.labels}))

# 提取模型并继续训练
train_step = sess.graph.get_operation_by_name('train/gd_train')
for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={input: batch_xs, valid: batch_ys})
print(sess.run(accuracy, feed_dict={
    input: mnist.test.images, valid: mnist.test.labels}))
