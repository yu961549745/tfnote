import tensorflow as tf


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


def build_graph():
    # 定义模型
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
            tf.nn.softmax_cross_entropy_with_logits(labels=output_valid, logits=output), name='cross_entropy')
        train_step = tf.train.AdamOptimizer(
            1e-4, name='train').minimize(cross_entropy)
        tf.summary.scalar('loss', cross_entropy)

    with tf.name_scope('valid'):
        correct_prediction = tf.equal(
            tf.argmax(output, 1), tf.argmax(output_valid, 1))
        accuracy = tf.reduce_mean(
            tf.cast(correct_prediction, tf.float32), name='accuracy')

    predict = tf.argmax(output, name='predict')

    return input_image, output_valid, keep_prob, train_step, accuracy, cross_entropy, predict


def restore_graph(sess):
    input_image = sess.graph.get_tensor_by_name('input_image:0')
    output_valid = sess.graph.get_tensor_by_name('valid_output:0')
    keep_prob = sess.graph.get_tensor_by_name('full_link/keep_prob:0')
    train_step = sess.graph.get_operation_by_name('train/train')
    accuracy = sess.graph.get_tensor_by_name('valid/accuracy:0')
    cross_entropy = sess.graph.get_tensor_by_name('train/cross_entropy:0')
    predict = sess.graph.get_tensor_by_name('predict:0')
    return input_image, output_valid, keep_prob, train_step, accuracy, cross_entropy, predict
