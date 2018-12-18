# package
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
        print('正确率  ' + str(correct / len(x)))
        return correct / len(x)


if __name__ == '__main__':
    girls = getData('../dataSet/female.txt')
    boys = getData('../dataSet/male.txt')
    one = fisher()
    one.fit(girls, boys)

    # x, y = getTest()
    x = numpy.r_[girls, boys]
    y = [0] * len(girls)
    for it in boys:
        y.append(1)
6
    one.score(x, y)
    b = []
    g = []
    cnt = 0
    for it in x:
        if(y[cnt] == 0):
            g.append(it)
        else:
            b.append(it)
        cnt = cnt + 1
    b = numpy.array(b)
    g = numpy.array(g)
    plt.scatter(g[:, 0], g[:, 1], color='red')
    plt.scatter(b[:, 0], b[:, 1], color='green')
    # plt.plot([0, 195], [one.w0, one.w0 + 195 * one.w[1] / one.w[0]])
    # plt.plot([120, 200], [one.w0 + 120 * one.w[1] / one.w[0],
    #                       one.w0 + 200 * one.w[1] / one.w[0]], color='green')
    # plt.plot([150, 195], [35 + one.w0, 35 + 45 *
    #                       one.w[0] / one.w[1]], color='green')
    plt.title('模式识别　fisher 分类器')
    plt.xlabel('身高')
    plt.ylabel('体重')
    plt.savefig('Fisher 分类器的决策边界')
    plt.show()

    # x = numpy.r_[girls, boys]
    # y = [0] * len(girls)
    # for it in boys:
    #     y.append(1)
    # cmap_light = ListedColormap(['#AAAAFF', '#AAFFAA', '#FFAAAA'])
    # ori_light = ListedColormap(['r', 'g'])
    # x1_min, x1_max = numpy.min(x[:, 0]) - 1, numpy.max(x[:, 0]) + 1
    # x2_min, x2_max = numpy.min(x[:, 1]) - 1, numpy.max(x[:, 1]) + 1
    # xx1, xx2 = numpy.meshgrid(numpy.arange(
    #     x1_min, x1_max, 0.1), numpy.arange(x2_min, x2_max, 0.1))
    # Z = one.predict_all(numpy.c_[xx1.ravel(), xx2.ravel()])
    # Z = Z.reshape(xx1.shape)
    # plt.figure()
    # # plt.pcolormesh(xx1, xx2, Z, cmap=cmap_light)
    # plt.contour(xx1, xx2, Z, cmap=ListedColormap(['blue']))

    # two = Bayers()
    # two.fit(girls, boys)
    # x1_min, x1_max = numpy.min(x[:, 0]) - 1, numpy.max(x[:, 0]) + 1
    # x2_min, x2_max = numpy.min(x[:, 1]) - 1, numpy.max(x[:, 1]) + 1
    # xx1, xx2 = numpy.meshgrid(numpy.arange(
    #     x1_min, x1_max, 0.1), numpy.arange(x2_min, x2_max, 0.1))
    # Z = two.predict_all(numpy.c_[xx1.ravel(), xx2.ravel()])
    # Z = Z.reshape(xx1.shape)
    # plt.contour(xx1, xx2, Z, cmap=ListedColormap(['r']))
    # plt.legend(['fisher 决策边界', 'Bayers 决策边界'])

    # plt.scatter(x[:, 0], x[:, 1], c=y, cmap=ori_light)
    # # plt.title('决策边界　Bayers(红色) VS fisher(蓝色)')
    # plt.title('fisher 判别器的决策决策边界')
    # plt.xlabel('身高')
    # plt.ylabel('体重')
    # plt.show()

    # 绘制与先验概率的关系
    # x, y = getTest()
    # accuracy = []
    # rate = 1e-2
    # while(rate <= 1):
    #     accuracy.append([rate, 1 - one.score(x, y, 1 - rate, rate)])
    #     rate += 0.01
    # accuracy = numpy.array(accuracy)
    # plt.plot(accuracy[:, 0], accuracy[:, 1])
    # plt.title('随着女生先验概率递增，错误率的变化曲线')
    # plt.xlabel('女生先验概率')
    # plt.ylabel('错误率')

    # one = Bayers()
    # one.fit(girls, boys)
    # accuracy = []
    # rate = 1e-2
    # while(rate <= 1):
    #     accuracy.append([rate, 1 - one.score(x, y, rate)])
    #     rate += 0.01
    # accuracy = numpy.array(accuracy)
    # plt.plot(accuracy[:, 0], accuracy[:, 1])
    # plt.legend(['Fisher', 'Bayers'])
    # plt.show()

    # # 留一法
    # rate = 1e-2
    # accuracy = []
    # while(rate <= 1):
    #     l = len(girls)
    #     r = len(boys)
    #     correct = 0
    #     for i in range(l):
    #         new_girls = []
    #         for j in range(l):
    #             if(j != i):
    #                 new_girls.append(girls[j])
    #         new_girls = numpy.array(new_girls)
    #         one.fit(new_girls, boys)
    #         if(one.predict(girls[i], rate) == 0):
    #             correct = correct + 1
    #     for i in range(r):
    #         new_boys = []
    #         for j in range(r):
    #             if(j != i):
    #                 new_boys.append(boys[j])
    #         new_boys = numpy.array(new_boys)
    #         one.fit(girls, new_boys)
    #         if(one.predict(boys[i], 1 - rate) == 1):
    #             correct = correct + 1
    #     print(rate)
    #     print('正确率　　:  ' + str(correct / (l + r)))

    #     accuracy.append([rate, 1 - correct / (l + r)])
    #     rate += 0.01
    # accuracy = numpy.array(accuracy)
    # plt.plot(accuracy[:, 0], accuracy[:, 1])
    # plt.title('留一法：错误率随着女生先验概率的变化曲线')
    # plt.legend(['正常测试', '留一法测试']) # plt.title('错误率随着女生先验概率的变化曲线')
    # plt.xlabel('女生先验概率')
    # plt.ylabel('错误率')
    # plt.show()

    # #混淆矩阵
    # x, y = getTest()
    # res = one.predict_all(x)
    # print(res)
    # ans = confusion_matrix(y, res)
    # print(ans)
    # plt.matshow(ans)
    # for i in range(2):
    #     for j in range(2):
    #         plt.text(i, j, "%0.2f" % (ans[i][j],), color='red',
    #                  fontsize=20, va='center', ha='center')
    # plt.show()

    # fprs, tprs, thresholds = roc_curve(y, res)
    # plt.plot(fprs, tprs, color='red')

    # one = Bayers()
    # one.fit(girls, boys)
    # res = one.predict_all(x)
    # fprs, tprs, thresholds = roc_curve(y, res)
    # plt.plot(fprs, tprs, color='blue')
    # plt.show()
