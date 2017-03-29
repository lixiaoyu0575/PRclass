import numpy as np
from sklearn import preprocessing
import random
trainDivide = 0.4
divideNum = int(trainDivide * 50)
typeDic = { #define class
    'Iris-setosa\n': 0,
    'Iris-versicolor\n': 1,
    'Iris-virginica\n': 2
}
LOOP_NUM_GETDATA = 10000
LOOP_NUM_GETACC = 200
W_NUM = 20
def loadData(path):
    allData = [[], [], []]  # 0: setosa, 1: versicolor, 2:virginica
    with open(path) as f:
        for line in f:
            lineContent = line.split(',')
            irisType = lineContent[4]
            index = typeDic.get(irisType)
            lineContent[4] = index
            allData[index].append(lineContent)
    f.close()
    return allData

def preprocess(class1A, class2A):
    for a in class1A:
        a[4] = 0
    for b in class2A:
        b[4] = 1
    class1A, class2A = np.array(class1A,"float32"), np.array(class2A, "float32")
    train_x, train_y, test_x, test_y = [], [], [], []
    fit_trans = preprocessing.MinMaxScaler().fit_transform
    indices = np.arange(len(class1A))
    np.random.shuffle(indices)
    indices1 = indices[0: divideNum]
    indices2 = indices[divideNum: 50]
    for index1 in indices1:
        train_x.append(class1A[index1][0:4])
        train_y.append(class1A[index1][4])
    for index1 in indices1:
        train_x.append(class2A[index1][0:4])
        train_y.append(class2A[index1][4])
    for index2 in indices2:
        test_x.append(class1A[index2][0:4])
        test_y.append(class1A[index2][4])
    for index2 in indices2:
        test_x.append(class2A[index2][0:4])
        test_y.append(class2A[index2][4])
    return fit_trans(np.array(train_x)), np.array(train_y), fit_trans(np.array(test_x)), np.array(test_y)

def getAccuracy(test_y, pre_y):
    count = 0
    l = len(test_y)
    for i in range(l):
        if test_y[i] == pre_y[i]:
            count += 1
    return float(count) / len(test_y)
def perceptron_train(train_x, train_y, W, iterate_acc, iterate_max, alpha, data_order="in_order"):
    #data_order: "in_order" or "random"
    iterate_err = 1
    iterate_count = 0
    train_x_length = len(train_x)
    loop_num = range(train_x_length)
    if data_order == "random":
        loop_num = loop_num * LOOP_NUM_GETDATA
    for i in loop_num:
        if data_order == "random":
            i = random.randint(0, train_x_length - 1)
        x = train_x[i]
        y = train_y[i]
        x.shape = (len(x), 1)
        x_trans = np.transpose(x)
        while iterate_count < iterate_max:
            pre_y = np.dot(x_trans, W)
            iterate_err = y - pre_y
            err = abs(iterate_err[0])
            if err < iterate_acc:
                # print "err<acc"
                break
            tao = 2 * alpha * iterate_err * x
            W = W + tao
            iterate_count += 1
            # print 1
    return W
def perceptron_test(test_x, W):
    pre_y = []
    for a in test_x:
        a.shape = (len(a), 1)
        x_trans = np.transpose(a)
        y = np.dot(x_trans, W)
        if y > 0.5:
            pre_y.append(1)
        else:
            pre_y.append(0)
    return pre_y

irisData = loadData("./../iris_data_set/iris.csv")
print "data preprocessed"
acc0_1 = 0
for i in range(LOOP_NUM_GETACC):
    train_x, train_y, test_x, test_y = preprocess(irisData[0], irisData[1])
    train_best_acc = 0
    for j in range(W_NUM):
        if train_best_acc == 1.0:
            break
        W = np.random.rand(4, 1)
        W = perceptron_train(train_x, train_y, W, 0.001, 10000, 0.0001)
        train_pre_y = perceptron_test(train_x, W)
        train_acc = getAccuracy(train_y, train_pre_y)
        if train_acc > train_best_acc:
            best_W = W
            train_best_acc = train_acc
            print "epoch ", i, ": better train_accuracy at ", j + 1, "Time: ", train_best_acc
    pre_y = perceptron_test(test_x, best_W)
    acc = getAccuracy(test_y, pre_y)
    acc0_1 += acc
    print "epoch ", i, ": best train_accuracy = ", train_best_acc, " test_accuracy = ", acc, "\n"
accuracy0_1 = acc0_1 / LOOP_NUM_GETACC
print "current train/test data divide scale: trainDivide = ", trainDivide
print "Finall perceptron accuracy for Iris: ", accuracy0_1