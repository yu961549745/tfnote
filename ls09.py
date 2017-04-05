"""
日志和检测

TensorFlow 具有5个级别的消息：
DEBUG
INFO
WARN 
ERROR 
FATAL 

具有以下几种监视器 Monitor
CaptureVariable 保存特定变量的值
PrintTensor 输出tensor的值
SummarySaver 使用tf.summary.FileWriter 保存 
ValidationMonitor 可以指定停止条件
"""

#%%
import os

import numpy as np
import tensorflow as tf

# 设置日志输出级别为 INFO
# 设置之后，TensorFlow 自动每隔 100 次训练输出一次 loss
tf.logging.set_verbosity(tf.logging.INFO)

# Data sets
IRIS_TRAINING = "iris/iris_training.csv"
IRIS_TEST = "iris/iris_test.csv"


def main(unused_argv):
    # Load datasets.
    training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
        filename=IRIS_TRAINING, target_dtype=np.int, features_dtype=np.float)
    test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
        filename=IRIS_TEST, target_dtype=np.int, features_dtype=np.float)

    validation_metrics = {
        "accuracy":
            tf.contrib.learn.MetricSpec(
                metric_fn=tf.contrib.metrics.streaming_accuracy,
                prediction_key="classes"),
        "precision":
            tf.contrib.learn.MetricSpec(
                metric_fn=tf.contrib.metrics.streaming_precision,
                prediction_key="classes"),
        "recall":
            tf.contrib.learn.MetricSpec(
                metric_fn=tf.contrib.metrics.streaming_recall,
                prediction_key="classes")
    }
    # 指定每50步进行一次验证
    validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
        test_set.data,
        test_set.target,
        every_n_steps=50,
        metrics=validation_metrics,
        early_stopping_metric="loss",  # 通过检测 loss 来判定提前结束
        early_stopping_metric_minimize=True,  # 表示最小化
        early_stopping_rounds=200)  # 最小化是表示连续几次不变小

    # Specify that all features have real-value data
    feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]

    # Build 3 layer DNN with 10, 20, 10 units respectively.
    classifier = tf.contrib.learn.DNNClassifier(
        feature_columns=feature_columns,
        hidden_units=[10, 20, 10],
        n_classes=3,
        model_dir="iris_model",
        # 因为验证依赖于checkpoints,而iris数据集很小，所以指定每1秒保存一次
        config=tf.contrib.learn.RunConfig(save_checkpoints_secs=1))

    # Fit model.
    classifier.fit(x=training_set.data,
                   y=training_set.target,
                   steps=2000,
                   monitors=[validation_monitor])

    # Evaluate accuracy.
    accuracy_score = classifier.evaluate(
        x=test_set.data, y=test_set.target)["accuracy"]
    print("Accuracy: {0:f}".format(accuracy_score))

    # Classify two new flower samples.
    new_samples = np.array(
        [[6.4, 3.2, 4.5, 1.5], [5.8, 3.1, 5.0, 1.7]], dtype=float)
    y = list(classifier.predict(new_samples))
    print("Predictions: {}".format(str(y)))


if __name__ == "__main__":
    tf.app.run()
