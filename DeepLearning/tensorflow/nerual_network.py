# -*- coding: utf-8 -*-

import numpy as np
import datetime
from tensorflow.examples.tutorials.mnist import input_data


class FullConnectedLayer(object):
    def __init__(self, input_size, output_size, activator):
        '''
        构造函数
        input_size: 本层输入向量的维度
        output_size: 本层输出向量的维度
        activator: 激活函数
        '''
        self.input_size = input_size
        self.output_size = output_size
        self.activator = activator
        # 权重数组W
        # self.W = np.random.uniform(-0.1, 0.1, (output_size, input_size))
        self.W = np.random.uniform(-0.1, 0.1, (input_size, output_size))
        # 偏置项b
        # self.b = np.zeros((output_size, 1))
        self.b = np.zeros((1, output_size))
        # 输出向量
        # self.output = np.zeros((output_size, 1))
        self.output = np.zeros((1, output_size))

    def forward(self, input_array):
        '''
        前向计算
        input_array: 输入向量，维度必须等于input_size
        '''
        self.input = input_array
        self.output = self.activator.forward(
            np.dot(input_array, self.W) + self.b)

    def backward(self, delta_array):
        '''
        反向计算W和b的梯度
        delta_array: 从上一层传递过来的误差项
        '''
        self.delta = self.activator.backward(self.input) * np.dot(
            delta_array, self.W.T)
        self.W_grad = np.dot(self.input.T, delta_array)
        self.b_grad = delta_array

    def update(self, learning_rate):
        '''
        使用梯度下降算法更新权重
        '''
        self.W += learning_rate * self.W_grad
        self.b += learning_rate * self.b_grad


class SigmoidActivator(object):
    '''
    Sigmoid激活函数类
    '''

    def forward(self, weighted_input):
        '''
        sigmoid 函数
        weighted_input = np.dot(W.T, x)
        '''
        return 1.0 / (1.0 + np.exp(-weighted_input))

    def backward(self, output):
        '''
        sigmoid的导数
        '''
        return output * (1 - output)


class Network(object):
    '''
    神经网络
    '''

    def __init__(self, layers):
        self.layers = []
        for i in range(len(layers) - 1):
            self.layers.append(
                FullConnectedLayer(layers[i], layers[i + 1],
                                   SigmoidActivator()))

    def predict(self, sample):
        '''
        使用神经网络实现预测
        sample: 输入样本
        '''
        output = sample
        for layer in self.layers:
            layer.forward(output)
            output = layer.output
        return output

    def train(self, labels, data_set, rate, epoch):
        '''
        训练函数
        labels: 样本标签
        data_set: 输入样本
        rate: 学习速率
        epoch: 训练轮数
        '''
        for i in range(epoch):
            for d in range(len(data_set)):
                self.train_one_sample(labels[d], data_set[d], rate)

    def train_one_sample(self, label, sample, rate):
        '''
        使用一个样本进行训练
        '''
        #将向量转换为二维数组，方便计算
        self.predict(sample.reshape(1, len(sample)))
        self.calc_gradient(label)
        self.update_weight(rate)

    def calc_gradient(self, label):
        '''
        计算delta
        '''
        delta = self.layers[-1].activator.backward(self.layers[-1].output) * (
            label - self.layers[-1].output)
        # 反向传播
        for layer in self.layers[::-1]:
            layer.backward(delta)
            delta = layer.delta
        return delta

    def update_weight(self, rate):
        '''
        更新权重
        '''
        for layer in self.layers:
            layer.update(rate)


def get_result(vec):
    max_value_index = 0
    max_value = 0
    for i in range(len(vec)):
        if vec[i] > max_value:
            max_value = vec[i]
            max_value_index = i
    return max_value_index


def evaluate(network, test_data_set, test_labels):
    error = 0
    total = len(test_data_set)

    for i in range(total):
        label = get_result(test_labels[i])
        predict = get_result(network.predict(test_data_set[i])[0])
        if label != predict:
            error += 1
    return float(error) / float(total)


def train_and_evaluate():
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    last_error_ratio = 1.0
    epoch = 0
    network = Network([784, 300, 10])
    while True:
        train_images, train_labels = mnist.train.next_batch(100)
        test_images, test_labels = mnist.test.next_batch(100)
        epoch += 1
        network.train(train_labels, train_images, 0.3, 1)
        print '%s epoch %d finished' % (datetime.datetime.now(), epoch)
        if epoch % 10 == 0:
            error_ratio = evaluate(network, test_images, test_labels)
            print '%s after epoch %d, error ratio is %f' % (
                datetime.datetime.now(), epoch, error_ratio)
            if error_ratio > last_error_ratio:
                break
            else:
                last_error_ratio = error_ratio


if __name__ == '__main__':
    train_and_evaluate()