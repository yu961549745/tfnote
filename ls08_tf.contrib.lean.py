#%% load data
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

# Data sets
IRIS_TRAINING = "./iris/iris_training.csv"
IRIS_TEST = "./iris/iris_test.csv"

# Load datasets.
# csv 文件读取方法
training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=IRIS_TRAINING,
    target_dtype=np.int,
    features_dtype=np.float32)
test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=IRIS_TEST,
    target_dtype=np.int,
    features_dtype=np.float32)

#%% train
# Specify that all features have real-value data
# 表示所有输入数据是4维实值向量
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]

# Build 3 layer DNN with 10, 20, 10 units respectively.
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,  # 指定特性
                                            # 指定隐含层神经元个数
                                            hidden_units=[10, 20, 10],
                                            n_classes=3,  # 指定输出层
                                            model_dir="./iris_model")  # 指定输出文件夹


def get_train_inputs():  # Define the training inputs
    x = tf.constant(training_set.data)
    y = tf.constant(training_set.target)
    return x, y


# Fit model.
classifier.fit(input_fn=get_train_inputs, steps=2000)


def get_test_inputs():  # Define the test inputs
    x = tf.constant(test_set.data)
    y = tf.constant(test_set.target)
    return x, y


#%% Evaluate accuracy.
accuracy_score = classifier.evaluate(input_fn=get_test_inputs,
                                     steps=1)["accuracy"]

print("\nTest Accuracy: {0:f}\n".format(accuracy_score))


def new_samples():  # Classify two new flower samples.
    return np.array(
        [[6.4, 3.2, 4.5, 1.5],
            [5.8, 3.1, 5.0, 1.7]], dtype=np.float32)


predictions = list(classifier.predict_classes(input_fn=new_samples))

print(
    "New Samples, Class Predictions:    {}\n"
    .format(predictions))
