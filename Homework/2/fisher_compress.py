# package
import time
import math
import numpy
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
# self


def getTest():
    test = []
    file = open('../dataSet/test2.txt')
    r = file.readlines()
    for line in r:
        line = line.replace('\n', '').replace(
            'F', '0').replace('M', '1').strip()
        line = line.split('\t')
        test.append(line)
    file.close()
    test = numpy.array(test, dtype=numpy.float)
    return test[:, :2], test[:, 2]


def getData(fileName):
    collect = []
    file = open(fileName)
    r = file.readlines()
    for line in r:
        line = line.replace('\n', '').strip()
        collect.append(line.split('\t'))
    file.close()
    collect = numpy.array(collect, dtype=numpy.float)
    return collect


class Bayers():
    def fit(self, girls, boys):
        self.mean_girls = numpy.mean(girls, axis=0)
        self.cov_girls = numpy.cov(girls.T)
        self.cov_girls[0][1] = self.cov_girls[1][0] = 0.0
        self.inv_girls = numpy.linalg.inv(self.cov_girls)
        self.det_girls = numpy.linalg.det(self.cov_girls)

        self.mean_boys = numpy.mean(boys, axis=0)
        self.cov_boys = numpy.cov(boys.T)
        self.cov_boys[0][1] = self.cov_boys[1][0] = 0.0
        self.inv_boys = numpy.linalg.inv(self.cov_boys)
        self.det_boys = numpy.linalg.det(self.cov_boys)

    def predict(self, inpt, rate=0.5):
        lhs = -0.5 * ((inpt - self.mean_girls).T.dot(self.inv_girls).dot(inpt -
                                                                         self.mean_girls) + math.log(self.det_girls)) + math.log(rate)
        rhs = -0.5 * ((inpt - self.mean_boys).T.dot(self.inv_boys).dot(inpt -
                                                                       self.mean_boys) + math.log(self.det_boys)) + math.log(1 - rate)
        return 0 if(lhs > rhs) else 1

    def predict_all(self, inpt):
        res = []
        for it in inpt:
            res.append(self.predict(it))
        return numpy.array(res)

    def score(self, x, y, rate=0.5):
        cnt = 0
        correct = 0
        for it in x:
            res = self.predict(it, rate)
            if(res == y[cnt]):
                correct = correct + 1
            cnt = cnt + 1
        print('正确率　　' + str(correct / len(x)))
        return correct / len(x)


class fisher():
    def fit(self, lhs, rhs):
        self.mean_lhs = numpy.mean(lhs, axis=0)
        cov_lhs = numpy.dot((lhs - self.mean_lhs).T, lhs - self.mean_lhs)
        self.mean_rhs = numpy.mean(rhs, axis=0)
        cov_rhs = numpy.dot((rhs - self.mean_rhs).T, rhs - self.mean_rhs)
        Sw = cov_lhs + cov_rhs
        self.inv_Sw = numpy.linalg.inv(Sw)
        self.w = self.inv_Sw.dot(self.mean_lhs - self.mean_rhs)
        self.w0 = -0.5 * (self.mean_lhs + self.mean_rhs).dot(self.inv_Sw).dot(self.mean_lhs -
                                                                              self.mean_rhs)

    def predict(self, inpt, lhs=0.5, rhs=0.5):
        self.w0 = -0.5 * (self.mean_lhs + self.mean_rhs).dot(self.inv_Sw).dot(self.mean_lhs -
                                                                              self.mean_rhs) - math.log(rhs / lhs)
        # w0 = - 0.5 * self.w.dot(self.mean_lhs + self.mean_rhs)
        res = self.w.dot(inpt) + self.w0
        return 0 if(res > 0) else 1

    def predict_all(self, inpt):
        res = []
        for it in inpt:
            res.append(self.predict(it))
        return numpy.array(res)

    def score(self, x, y, lhs=0.5, rhs=0.5):
        x, y = getTest()
        cnt = 0
        correct = 0
        for it in x:
            if(self.predict(it, lhs, rhs) == y[cnt]):
                correct += 1
            cnt = cnt + 1
        # print('正确率  ' + str(correct / len(x)))
        return correct / len(x)


def make():
    x = numpy.r_[girls, boys]
    y = [0] * len(girls)
    for it in boys:
        y.append(1)
    x1_min, x1_max = numpy.min(x[:, 0]) - 1, numpy.max(x[:, 0]) + 1
    x2_min, x2_max = numpy.min(x[:, 1]) - 1, numpy.max(x[:, 1]) + 1
    xx1, xx2 = numpy.meshgrid(numpy.arange(
        x1_min, x1_max, 0.1), numpy.arange(x2_min, x2_max, 0.1))
    Z = one.predict_all(numpy.c_[xx1.ravel(), xx2.ravel()])
    Z = Z.reshape(xx1.shape)
    plt.figure()
    plt.contour(xx1, xx2, Z, cmap=ListedColormap(['blue']))
    plt.scatter(x[:, 0], x[:, 1], c=y, cmap=ListedColormap(['r', 'g']))
    plt.legend(['女生', '男生'])
    plt.title('剪辑之后')
    plt.xlabel('身高')
    plt.ylabel('体重')
    plt.show()


if __name__ == '__main__':
    girls = getData('../dataSet/female.txt')
    boys = getData('../dataSet/male.txt')
    one = fisher()
    one.fit(girls, boys)
    x, y = getTest()
    print('剪辑之前在测试集上的正确率  :  ' + str(one.score(x, y)))
    make()

    x = numpy.r_[girls, boys]
    y = [0] * len(girls)
    for it in boys:
        y.append(1)
    while(True):
        cnt = 0
        bad = []
        for it in x:
            res = one.predict(it)
            if(res != y[cnt]):
                bad.append(cnt)
            cnt = cnt + 1
        if(any(bad) == False):
            break
        print(bad)
        y = numpy.delete(y, bad, 0)
        res = []
        for it in y:
            if(it == 0):
                res.append(it)
        l = len(res)
        x = numpy.delete(x, bad, 0)
        girls = x[:l]
        boys = x[l:]
        one.fit(girls, boys)
        # break

    cnt = 0
    correct = 0
    for it in x:
        if(one.predict(it) == y[cnt]):
            correct += 1
        cnt = cnt + 1
    print('剪辑后训练集的正确率 ' + str(correct / len(x)))

    x, y = getTest()
    print('剪辑后在测试集上的正确率  :  ' + str(one.score(x, y)))
    make()
