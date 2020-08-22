#!/usr/local/bin/python3
# coding: utf-8

import sys
import numpy as np
import pandas as pd
from KNearestNeighbor import KNearestNeighbor
from NeuralNetwork import NeuralNetwork
from DecisionTree import DecisionTree


def read_data(fname):
    data = []
    file = open(fname, 'r')
    for line in file:
        data.append(line.split())
    return np.array(data)


def train_nearest_neighbor(train_x, train_y, model_file):
    knn = KNearestNeighbor(no_of_neighbors=55)
    knn.fit(train_x, train_y, batch_size=1000)
    print('training complete')
    knn.save(model_file)
    print('model saved successfully')


def test_nearest_neighbor(test_x, test_y, model_file):
    knn = KNearestNeighbor()
    knn.load(model_file)
    print('model loaded successfully')
    score = knn.score(test_x, test_y)
    print('Testing Accuracy: ', score)


def train_neural_network(train_x, train_y, model_file):
    nn = NeuralNetwork()
    nn.add(shape=(train_x.shape[1], train_x.shape[1]), activation='relu')
    nn.add(shape=(train_x.shape[1], train_x.shape[1]), activation='relu')
    nn.add(shape=(train_x.shape[1], train_x.shape[1]), activation='relu')
    nn.add(shape=(train_x.shape[1], 4), activation='softmax')

    epoch = 100
    nn.fit(train_x, train_y, batch_size=1, epoch=epoch, learning_rate=0.0005)
    print('training complete')
    nn.save(model_file)
    print('model saved successfully')


def test_neural_network(test_x, test_y, model_file):
    nn = NeuralNetwork()
    nn.load(model_file)
    print('model loaded successfully')
    print('Testing Accuracy: ', nn.evaluate(test_x, test_y))


def train_decision_tree(train_x, train_y, model_file):
    dt = DecisionTree(max_depth=10, minimum_num_leaves=10)
    dt.fit(train_x, train_y)
    print('training complete')
    dt.save(model_file)
    print('model saved successfully')


def test_decision_tree(test_x, test_y, model_file):
    dt = DecisionTree()
    dt.load(model_file)
    print('model loaded successfully')
    print('Testing Accuracy: ', dt.score(test_x, test_y))


def one_hot_encoding(y):
    y[y == 90] = 1
    y[y == 180] = 2
    y[y == 270] = 3
    y = pd.get_dummies(y).values
    return y


if __name__ == '__main__':

    if len(sys.argv) != 5:
        raise Exception("usage: ./orient.py train train_file.txt model_file.txt [model]")

    function, data_file, model_file, model = sys.argv[1:]

    if function == 'train':
        train_data = read_data(data_file)
        train_x = np.array(train_data[:, 2:], dtype='float')
        train_y = np.array(train_data[:, 1], dtype='int')

        if model == 'nearest':
            train_y = train_y.reshape(train_y.shape[0], 1)
            train_nearest_neighbor(train_x, train_y, model_file)

        elif model == 'nnet':
            train_x /= 255
            train_y = one_hot_encoding(train_y)
            train_neural_network(train_x, train_y, model_file)

        elif model == 'tree':
            train_y = train_y.reshape(train_y.shape[0], 1)
            train_decision_tree(train_x, train_y, model_file)
        else:
            train_x /= 255
            train_y = one_hot_encoding(train_y)
            train_neural_network(train_x, train_y, model_file)

    elif function == 'test':
        test_data = read_data(data_file)
        test_x = np.array(test_data[:, 2:], dtype='float')
        test_y = np.array(test_data[:, 1], dtype='int')

        if model == 'nearest':
            test_y = test_y.reshape(test_y.shape[0], 1)
            test_nearest_neighbor(test_x, test_y, model_file)
        elif model == 'nnet':
            test_x /= 255
            test_y = one_hot_encoding(test_y)
            test_neural_network(test_x, test_y, model_file)
        elif model == 'tree':
            test_y = test_y.reshape(test_y.shape[0], 1)
            test_decision_tree(test_x, test_y, model_file)
        else:
            test_x /= 255
            test_y = one_hot_encoding(test_y)
            test_neural_network(test_x, test_y, model_file)
    else:
        print('something went wrong !!!')
        print('usage: ./orient.py train train_file.txt model_file.txt [model]')
        print('usage: ./orient.py test test_file.txt model_file.txt [model]')
