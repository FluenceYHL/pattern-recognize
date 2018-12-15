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

    def predict(self, inpt):
        lhs = -0.5 * ((inpt - self.mean_girls).T.dot(self.inv_girls).dot(inpt -
                                                                         self.mean_girls) + math.log(self.det_girls)) + math.log(0.2)
        rhs = -0.5 * ((inpt - self.mean_boys).T.dot(self.inv_boys).dot(inpt -
                                                                       self.mean_boys) + math.log(self.det_boys)) + math.log(0.8)
        return 0 if(lhs > rhs) else 1

    def predict_all(self, inpt):
        res = []
        cnt = 0
        for it in inpt:
            res.append(self.predict(it))
            if(cnt % 10000 == 0):
                print(str(cnt / 10000) + ' is over.....')
            cnt = cnt + 1
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


if __name__ == '__main__':
    girls = getData('../dataSet/female.txt')
    boys = getData('../dataSet/male.txt')
    one = Bayers()
    one.fit(girls, boys)
    x, y = getTest()
    one.score(x, y)

    cmap_light = ListedColormap(['#AAAAFF', '#AAFFAA', '#FFAAAA'])
    ori_light = ListedColormap(['r', 'g'])
    # 获取边界范围, 为了产生数据
    x1_min, x1_max = numpy.min(x[:, 0]) - 1, numpy.max(x[:, 0]) + 1
    x2_min, x2_max = numpy.min(x[:, 1]) - 1, numpy.max(x[:, 1]) + 1

    # 生成新的数据, 并调用meshgrid网格搜索函数帮助我们生成矩阵
    xx1, xx2 = numpy.meshgrid(numpy.arange(
        x1_min, x1_max, 0.1), numpy.arange(x2_min, x2_max, 0.1))
    # 有了新的数据, 我们需要将这些数据输入到分类器获取到结果, 但是因为输入的是矩阵, 我们需要给你将其转换为符合条件的数据
    Z = one.predict_all(numpy.c_[xx1.ravel(), xx2.ravel()])
    # 这个时候得到的是Z还是一个向量, 将这个向量转为矩阵即可
    Z = Z.reshape(xx1.shape)
    plt.figure()
    # 分解的时候有背景颜色
    plt.pcolormesh(xx1, xx2, Z, cmap=cmap_light)
    # 为什么需要输入矩阵, 因为等高线函数其实是3D函数, 3D坐标是三个平面, 平面对应矩阵
    plt.contour(xx1, xx2, Z, cmap=plt.cm.RdYlBu)
    plt.scatter(x[:, 0], x[:, 1], c=y, cmap=ori_light)
    plt.title('Bayers 分类器决策边界')
    plt.xlabel('身高')
    plt.ylabel('体重')
    plt.savefig('Bayers 分类器决策边界.png')
    plt.show()
