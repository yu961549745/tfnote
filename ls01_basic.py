""" TensorFlow 基础概念 """

#%% 导入 TensorFlow
import tensorflow as tf

#%% 什么是Tensor
# Tensor 是 TensorFlow 的基本对象
# 说白了就是多维向量
t0 = tf.constant(1)  # 0阶 tensor
t1 = tf.constant([1, 2])  # 1阶 tensor
t2 = tf.constant([[1, 2], [3, 4]])  # 2阶 tensor
t3 = tf.constant([[[1., 2., 3.]], [[7., 8., 9.]]])  # 3阶 tensor
print(t0)
print(t1)
print(t2)
print(t3)

#%% Session
# TensorFlow 的基本对象是 graph node  需要依赖于  Session 进行求值
sess = tf.Session()
print(sess.run([t0, t1, t2, t3]))

#%% 基本运算也是一个 graph node, 并且这些运算是向量化的
add = tf.add(t0, t1)
print(sess.run(add))

#%% placeholder
# 用来表示一个输入数据的占位符，其值在执行时给定
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
add_node = a + b
print(sess.run(add_node, {a: 2, b: 3}))
print(sess.run(add_node, {a: [1, 2], b: [3, 4]}))

#%% Variable
W = tf.Variable([1.], tf.float32)
b = tf.Variable([1.], tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b
sess.run(tf.global_variables_initializer())  # 必须显式声明初始化
print(sess.run(linear_model, {x: [1, 2, 3, 4]}))

#%% 定义损失函数
y = tf.placeholder(tf.float32)
loss = tf.reduce_sum(tf.square(linear_model - y))
print(sess.run(loss, {x: [1, 2, 3], y: [2, 4, 8]}))

#%% 赋值
sess.run([tf.assign(W, [2]), tf.assign(b, [-1])])
print(sess.run(loss, {x: [1, 2, 3], y: [1, 3, 5]}))

#%% 训练模型
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
sess.run(tf.global_variables_initializer())  # 重置为错误的值
# 训练
for i in range(1000):
    sess.run(train, {x: [1, 2, 3], y: [1, 3, 5]})

print(sess.run([W, b]))
