# package
import math
import numpy
import matplotlib.pyplot as plt
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
    return collect, numpy.mean(collect, axis=0)


if __name__ == '__main__':
    girls, mean_girls = getData('../dataSet/female.txt')
    cov_girls = numpy.cov(girls.T)
    cov_girls[0][1] = cov_girls[1][0] = 0.0
    inv_girls = numpy.linalg.inv(cov_girls)
    det_girls = numpy.linalg.det(cov_girls)

    boys, mean_boys = getData('../dataSet/male.txt')
    cov_boys = numpy.cov(boys.T)
    cov_boys[0][1] = cov_boys[1][0] = 0.0
    inv_boys = numpy.linalg.inv(cov_boys)
    det_boys = numpy.linalg.det(cov_boys)

    # x, y = getTest()
    # accuracy = []
    # rate = 1e-2
    # while(rate <= 1.00):
    #     cnt = 0
    #     correct = 0
    #     for it in x:
    #         lhs = -0.5 * ((it - mean_girls).T.dot(inv_girls).dot(it -
    #                                                              mean_girls) + math.log(det_girls)) + math.log(rate)
    #         rhs = -0.5 * ((it - mean_boys).T.dot(inv_boys).dot(it -
    #                                                            mean_boys) + math.log(det_boys)) + math.log(1 - rate)
    #         res = 0 if(lhs > rhs) else 1
    #         if(res == y[cnt]):
    #             correct = correct + 1
    #         cnt = cnt + 1
    #     rate += 0.01
    #     accuracy.append([rate, correct / len(x)])
    # accuracy = numpy.array(accuracy)
    # x2 = accuracy[:, 0]
    # y2 = accuracy[:, 1]
    # plt.plot(x2, y2)
    # plt.title('模式识别　1-Bayers 分类器')
    # plt.xlabel('男生先验概率')
    # plt.ylabel('正确率')
    # plt.show()

    x, y = getTest()

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

    plt.scatter(g[:, 0], g[:, 1])
    plt.scatter(b[:, 0], b[:, 1])
    plt.legend(['女生', '男生'])
    plt.title('模式识别　1-Bayers 分类器')
    plt.xlabel('男生先验概率')
    plt.ylabel('正确率')
    plt.show()
