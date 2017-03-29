import numpy as np
from sklearn import preprocessing
import random
import math
import matplotlib.pyplot as plt
trainDivide = 0.8
divideNum = int(trainDivide * 50)
typeDic = { #define class
    'Iris-setosa\n': 0,
    'Iris-versicolor\n': 1,
    'Iris-virginica\n': 2
}
LOOP_NUM_GETDATA = 10000
LOOP_NUM_GETACC = 200
BATCH_NUM = 10
EPOCH_NUM = 100000

def getAccuracy(test_y, pre_y):
    count = 0
    l = len(test_y)
    for i in range(l):
        if test_y[i] == pre_y[i]:
            count += 1
    return float(count) / len(test_y)

def sigmoid(x,derivate=False):
    if derivate:
        return sigmoid(x)*(1-sigmoid(x))
    return 1.0 / (1+math.exp(-x))

def train_nn(train_x, train_y, W1, W2, iterate_acc, iterate_max, alpha):
    iterate_err = 1
    iterate_count = 0
    train_x_length = len(train_x)
    loop_num = range(train_x_length)
    train_acc_record = []
    lowest_err = 1
    for k in range(EPOCH_NUM):
        indices = np.arange(train_x_length)
        np.random.shuffle(indices)
        train_indices = indices[0: BATCH_NUM - 1]
        train_indices = indices[0: train_x_length - 1]
        for i in train_indices:
            x, y = train_x[i], train_y[i]
            x.shape = (len(x), 1)
            x_trans = np.transpose(x)
            out1 = np.dot(x_trans, W1)
            for j in range(len(out1[0])):
                out1[0][j] = sigmoid(out1[0][j])
            pre_y_before_sig = np.dot(out1, W2)
            pre_y = sigmoid(pre_y_before_sig)
            iterate_err = y - pre_y
            err = abs(iterate_err)
            if err < lowest_err:
                lowest_err = err
            # if err < iterate_acc:
            #     print "err<acc"
            #     train_acc_record = np.array(train_acc_record)
            #     plt.plot(train_acc_record[:, 0], train_acc_record[:, 1], '*')
            #     plt.show()
            #     return W1, W2, lowest_err
            tao_o = 2 * alpha * iterate_err * out1 * pre_y * (1 - pre_y)
            W2 += np.transpose(tao_o)
            test = out1 * (1 - out1)
            # print 2
            tao_h = alpha * np.dot(tao_o, W2) * x * out1 * (1 - out1)
            W1 += tao_h
            # print iterate_err
        train_pre_y = test_nn(train_x, W1, W2)
        train_acc = getAccuracy(train_y, train_pre_y)
        print "Epoch ", k+1, " training accuracy = ", train_acc
        train_acc_record.append([k+1, train_acc])
        if train_acc > 0.99:
            break
    train_acc_record = np.array(train_acc_record)
    plt.plot(train_acc_record[:, 0], train_acc_record[:, 1], '*')
    plt.show()
    return W1, W2, lowest_err

def test_nn(test_x, W1, W2):
    pre_y = []
    for x in test_x:
        x.shape = (len(x), 1)
        x_trans = np.transpose(x)
        out1 = np.dot(x_trans, W1)
        for j in range(len(out1[0])):
            out1[0][j] = sigmoid(out1[0][j])
        y_before_sig = np.dot(out1, W2)
        y = sigmoid(y_before_sig)
        if y > 0.5:
            pre_y.append(0.75)
        else:
            pre_y.append(0.25)
    return pre_y


XOR_data = np.loadtxt("./generated_XOR_data.txt")
# preprocess data
XOR_data_length = len(XOR_data)
train_test_bound = int(XOR_data_length * trainDivide)
indices = np.arange(XOR_data_length)
np.random.shuffle(indices)
indices_train, indices_test = indices[0: train_test_bound], indices[train_test_bound: XOR_data_length - 1]
train_data, test_data = XOR_data[indices_train], XOR_data[indices_test]
fit_trans = preprocessing.MinMaxScaler().fit_transform
train_x, train_y = fit_trans(train_data[:, [0, 1]]), train_data[:, 2]
test_x, test_y = fit_trans(test_data[:, [0, 1]]), test_data[:, 2]

unit_num = 50
learning_rate = 0.1
W1 = np.random.rand(2, unit_num)
W2 = np.random.rand(unit_num * 1, 1)
W1, W2, lowest_err = train_nn(train_x, train_y, W1, W2, 0.0001, 1000000, 0.1)
pre_y = test_nn(test_x, W1, W2)
acc = getAccuracy(test_y, pre_y)
print "learning rate (alpha): ", learning_rate
print "hidden units num: ", unit_num
print "The lowest error: ", lowest_err
print "The final test accuracy: ", acc