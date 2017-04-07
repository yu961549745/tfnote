import tensorflow as tf
import ls05_mnist as model
import numpy as np

version = 1


class MNIST(object):
    def __init__(self):
        sess = tf.Session()
        saver = tf.train.import_meta_graph('MNIST_conv/conv.meta')
        saver.restore(sess, 'MNIST_conv/conv-999')
        input_image, _, keep_prob, _, _, _, output = model.restore_graph(
            sess)
        self.sess = sess
        self.input_image = input_image
        self.keep_prob = keep_prob
        self.output = output

    def predict(self, img):
        return self.sess.run(tf.argmax(self.output, 1), feed_dict={self.input_image: np.array(img, dtype=np.float32).reshape(-1, 784), self.keep_prob: 1.0})

    def close(self):
        self.sess.close()
