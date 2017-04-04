""" TensorFlow 高层训练接口 """
#%% tf.contrib.lean 高层训练接口，包含：
# 执行训练
# 执行求值
# 管理数据集
# 管理feeding

import tensorflow as tf
# NumPy 经常被用来生成和处理数据
import numpy as np

# Declare list of features. We only have one real-valued feature. There are many
# other types of columns that are more complicated and useful.
# 定义一系列的特征
features = [tf.contrib.layers.real_valued_column("x", dimension=1)]

# An estimator 用于拟合和求值
# TF提供了许多内置的 estimator:
# linear regression,
# logistic regression,
# linear classification,
# logistic classification,
# and many neural network classifiers and regressors.
estimator = tf.contrib.learn.LinearRegressor(feature_columns=features)

# TensorFlow provides many helper methods to read and set up data sets.
# Here we use `numpy_input_fn`. We have to tell the function how many batches
# of data (num_epochs) we want and how big each batch should be.
x = np.array([1., 2., 3., 4.])
y = np.array([0., -1., -2., -3.])
# 其实这里的 num_epochs 不是很明白
input_fn = tf.contrib.learn.io.numpy_input_fn({"x": x}, y, batch_size=4,
                                              num_epochs=1000)

# We can invoke 1000 training steps by invoking the `fit` method and passing the
# training data set.
estimator.fit(input_fn=input_fn, steps=1000)

# Here we evaluate how well our model did. In a real example, we would want
# to use a separate validation and testing data set to avoid overfitting.
print(estimator.evaluate(input_fn=input_fn))
