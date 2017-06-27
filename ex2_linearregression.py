# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 21:03:21 2017

@author: Alphatao
#线性回归 http://www.shareditor.com/
"""

import numpy as np
import tensorflow as tf

# 随机生成1000个点，范围在y=0.5x+0.5的直线附近
num_points = 1000
vectors_set = []
for i in range(num_points):
    x1 = np.random.normal(0.0, 0.55) 
    y1 = x1 * 0.5 + 0.5 + np.random.normal(0.0, 0.03)
    vectors_set.append([x1, y1])

# 生成样本集
x_data = [v[0] for v in vectors_set]
y_data = [v[1] for v in vectors_set]


# 生成1维的W矩阵，取值是[-1,1]之间的随机数
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0), name='W')

# 生成1维的偏差b矩阵，初始值是0
b = tf.Variable(tf.zeros([1]), name='b')

# 预测值y计算
y = W * x_data + b

# 以预测值y和实际值y_data之间的均方差作为损失值
loss = tf.reduce_mean(tf.square(y - y_data), name='loss')
# 采用梯度下降法来优化参数
optimizer = tf.train.GradientDescentOptimizer(0.5)
# 训练的过程就是最小化这个误差值
train = optimizer.minimize(loss, name='train')

sess = tf.Session()



#初始化
init = tf.initialize_all_variables()
sess.run(init)

# 初始化的W和b值
print ("W =", sess.run(W), "b =", sess.run(b), "loss =", sess.run(loss))

# 执行20次训练 输出每次的参数值
for step in range(20):
    sess.run(train)
    print ("W =", sess.run(W), "b =", sess.run(b), "loss =", sess.run(loss))




