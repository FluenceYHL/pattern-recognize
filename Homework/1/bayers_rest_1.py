# package
import math
import numpy
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
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

    def score(self, x, y):
        cnt = 0
        correct = 0
        for it in x:
            res = self.predict(it)
            if(res == y[cnt]):
                correct = correct + 1
            cnt = cnt + 1
        print('正确率　　' + str(correct / len(x)))
        return correct / len(x)


if __name__ == '__main__':
    girls = getData('../dataSet/female.txt')
    boys = getData('../dataSet/male.txt')
    one = Bayers()
    one.fit(girls, boys)
    x, y = getTest()
    one.score(x, y)

    l = len(girls)
    r = len(boys)
    correct = 0
    for i in range(l):
        new_girls = []
        for j in range(l):
            if(j != i):
                new_girls.append(girls[j])
        new_girls = numpy.array(new_girls)
        one.fit(new_girls, boys)
        if(one.predict(girls[i]) == 0):
            correct = correct + 1

    for i in range(r):
        new_boys = []
        for j in range(r):
            if(j != i):
                new_boys.append(boys[j])
        new_boys = numpy.array(new_boys)
        one.fit(girls, new_boys)
        if(one.predict(boys[i]) == 1):
            correct = correct + 1
    print('正确率　　:  ' + str(correct / (l + r)))
