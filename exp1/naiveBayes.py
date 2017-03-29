import numpy as np
trainDivide = 0.8
divideNum = int(trainDivide * 50)
typeDic = { #define class
    'Iris-setosa\n': 0,
    'Iris-versicolor\n': 1,
    'Iris-virginica\n': 2
}
batch_num = 10
feature_num = 4
acc0_1, acc1_2, acc0_2 = 0, 0, 0
acc0_1_feature, acc1_2_feature, acc0_2_feature = [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]

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
    train_x, train_y, test_x, test_y = [], [], [], []
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
    return np.array(train_x), np.array(train_y), np.array(test_x), np.array(test_y)

def getMeanAndVar(arr, index):
    arr_sum = 0
    for a in arr:
        arr_sum += float(a[index])
    mean = arr_sum / len(arr)
    dsum = 0
    for a in arr:
        d = float(a[index]) - mean
        dsum += (d * d)
    var = dsum / len(arr)
    return mean, var

def gaussion(x, mean, var):
    x = float(x)
    return np.exp(-(x - mean) * (x - mean) / 2 / var / var) / var

def classify(class1, class2, feature):
    mean1, var1 = getMeanAndVar(train_x[0: divideNum], feature)
    mean2, var2 = getMeanAndVar(train_x[divideNum: 2 * divideNum], feature)
    pre_y = []
    for a in test_x:
        prob1 = gaussion(a[feature], mean1, var1)
        prob2 = gaussion(a[feature], mean2, var2)
        if prob1 > prob2:
            pre_y.append(class1)
        else:
            pre_y.append(class2)
    return pre_y

def classify_all(class1, class2):
    mean11, var11 = getMeanAndVar(train_x[0: divideNum], 0)
    mean12, var12 = getMeanAndVar(train_x[0: divideNum], 1)
    mean13, var13 = getMeanAndVar(train_x[0: divideNum], 2)
    mean14, var14 = getMeanAndVar(train_x[0: divideNum], 3)
    mean21, var21 = getMeanAndVar(train_x[divideNum: 2 * divideNum], 0)
    mean22, var22 = getMeanAndVar(train_x[divideNum: 2 * divideNum], 1)
    mean23, var23 = getMeanAndVar(train_x[divideNum: 2 * divideNum], 2)
    mean24, var24 = getMeanAndVar(train_x[divideNum: 2 * divideNum], 3)
    pre_y = []
    for a in test_x:
        prob11 = gaussion(a[0], mean11, var11)
        prob12 = gaussion(a[1], mean12, var12)
        prob13 = gaussion(a[2], mean13, var13)
        prob14 = gaussion(a[3], mean14, var14)
        prob21 = gaussion(a[0], mean21, var21)
        prob22 = gaussion(a[1], mean22, var22)
        prob23 = gaussion(a[2], mean23, var23)
        prob24 = gaussion(a[3], mean24, var24)
        prob1 = prob11 * prob12 * prob13 * prob14
        prob2 = prob21 * prob22 * prob23 * prob24
        if prob1 > prob2:
            pre_y.append(class1)
        else:
            pre_y.append(class2)
    return pre_y

def getAccuracy(test_y, pre_y):
    count = 0
    l = len(test_y)
    for i in range(l):
        if test_y[i] == pre_y[i]:
            count += 1
    return float(count) / len(test_y)


irisData = loadData("./../iris_data_set/iris.csv")

for i in range(batch_num):
    train_x, train_y, test_x, test_y = preprocess(irisData[0], irisData[1])
    for j in range(feature_num):
        acc0_1_feature[j] += getAccuracy(test_y, classify(0, 1, j))
    acc0_1 += getAccuracy(test_y, classify_all(0, 1))

    train_x, train_y, test_x, test_y = preprocess(irisData[1], irisData[2])
    for j in range(feature_num):
        acc1_2_feature[j] += getAccuracy(test_y, classify(1, 2, j))
    acc1_2 += getAccuracy(test_y, classify_all(1, 2))

    train_x, train_y, test_x, test_y = preprocess(irisData[0], irisData[2])
    for j in range(feature_num):
        acc0_2_feature[j] += getAccuracy(test_y, classify(0, 2, j))
    acc0_2 += getAccuracy(test_y, classify_all(0, 2))

print "accuracy in 4-d between class 0 and 1: ", acc0_1 / batch_num
print "accuracy in 4-d between class 1 and 2: ", acc1_2 / batch_num
print "accuracy in 4-d between class 0 and 2: ", acc0_2 / batch_num
print "accuracy in each dimension between class 0 and 1: ", acc0_1_feature[0] / batch_num, acc0_1_feature[1] / batch_num, acc0_1_feature[2] / batch_num, acc0_1_feature[3] / batch_num
print "accuracy in each dimension between class 1 and 2: ", acc1_2_feature[0] / batch_num, acc1_2_feature[1] / batch_num, acc1_2_feature[2] / batch_num, acc1_2_feature[3] / batch_num
print "accuracy in each dimension between class 0 and 2: ", acc0_2_feature[0] / batch_num, acc0_2_feature[1] / batch_num, acc0_2_feature[2] / batch_num, acc0_2_feature[3] / batch_num
