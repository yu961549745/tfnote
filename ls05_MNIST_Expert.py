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
batchSize = 50
trainSteps = 500
logPeriod = 100
savePeriod = 1000
startStep = 0

# 读取数据
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(dataPath, one_hot=True)


import tensorflow as tf
import ls05_mnist as model


sess = tf.Session()
if startStep == 0:
    input_image, output_valid, keep_prob, train_step, accuracy, cross_entropy, _ = model.build_graph()

    # 第一次保存时清空现有文件夹
    if os.path.exists(modelSavePath):
        shutil.rmtree(modelSavePath)
    os.mkdir(modelSavePath)

    sess.run(tf.global_variables_initializer())
    summary = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(modelSavePath, sess.graph)
    saver = tf.train.Saver()
    saver.export_meta_graph(modelMetaFile)
else:
    saver = tf.train.import_meta_graph(modelMetaFile)
    saver.restore(sess, modelCkpPath + '-' + str(startStep - 1))
    input_image, output_valid, keep_prob, train_step, accuracy, cross_entropy, _ = model.restore_graph(
        sess)
    summary = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(modelSavePath, sess.graph)

# 训练
st = time.time()
for step in range(startStep, startStep + trainSteps):
    batch = mnist.train.next_batch(batchSize)
    if step % logPeriod == 0 or step == trainSteps - 1:
        _, loss_value, summary_str = sess.run([train_step, cross_entropy, summary], feed_dict={
            input_image: batch[0], output_valid: batch[1], keep_prob: 0.5})
        print("step = %d, loss = %g, time=%.3f sec" %
              (step, loss_value, time.time() - st))
        summary_writer.add_summary(summary_str, step)
        summary_writer.flush()
    else:
        sess.run(train_step, feed_dict={
            input_image: batch[0], output_valid: batch[1], keep_prob: 0.5})
    if (step + 1) % savePeriod == 0 or step == trainSteps - 1:
        savepath = saver.save(sess, modelCkpPath, global_step=step)
        print("save check point in %s" % (savepath))
print("test accuracy %g" % sess.run(accuracy, feed_dict={
    input_image: mnist.test.images, output_valid: mnist.test.labels, keep_prob: 1.0}))
sess.close()
