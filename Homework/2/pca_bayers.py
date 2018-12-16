# package
import math
import numpy
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
# self
from sklearn.decomposition import PCA


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


class YHL_pca():
    def __init__(self, percentage=0.998):
        self.percentage = percentage

    # 根据比例, 提取出前　K 个特征向量, 本函数返回　k
    def __getK(self, feature_values):
        res = 0
        index = 0
        features_sorted = numpy.sort(-feature_values)  # 从大到小排序
        sum_value = sum(feature_values)
        for it in feature_values:
            index += 1
            res += it
            if(res >= sum_value * self.percentage):
                return index

    def pca(self, features, percentage=0.998, k=0):
        self.percentage = percentage               # 更新本次　PCA 的比率
        mean_value = numpy.mean(features, axis=0)
        features = features - mean_value           # 减去均值, 更方便计算协方差
        cov_matrix = numpy.cov(features, rowvar=0)  # 求协方差矩阵
        feature_values, feature_vectors = numpy.linalg.eig(
            numpy.mat(cov_matrix))  # 求特征值，特征向量
        if(k == 0):
            k = self.__getK(feature_values)        # 如果没有指定　k, 就按照特征比率提取
        indexs = numpy.argsort(-feature_values)    # 对特征值从大到小排序，返回索引
        k_indexs = indexs[:k]                      # 提取　k 个特征值
        primary_components = feature_vectors[:, k_indexs]  # 找出对应的　k 个特征向量
        primary_components = abs(primary_components)
        return k, mean_value, primary_components, features.dot(primary_components)


def after_pro(x, mean_value, var_value, pre):
    gaussi = 1 / (math.sqrt(2 * math.pi * var_value))
    gaussi = gaussi * \
        math.exp((-1 * (x - mean_value) * (x - mean_value) / (2 * var_value)))
    return gaussi * pre


if __name__ == '__main__':
    girls = getData('../dataSet/female.txt')
    boys = getData('../dataSet/male.txt')
    train_data = numpy.r_[girls, boys]
    y = numpy.r_[[0] * len(girls), [1] * len(boys)]

    one = YHL_pca()
    k, mean_value, primary_components, new_x = one.pca(
        train_data, percentage=0.95, k=1)

    new_girls = new_x[:len(girls)]
    mean_girls = numpy.mean(new_girls)
    var_girls = numpy.var(new_girls)
    print(mean_girls)
    print(var_girls)

    new_boys = new_x[len(boys):]
    mean_boys = numpy.mean(new_boys)
    var_boys = numpy.var(new_boys)
    print(mean_boys)
    print(var_boys)

    x, y = getTest()
    x = x - numpy.mean(x, axis=0)
    x = x.dot(primary_components)
    x.resize(x.shape[:1])
    print(x.shape)

    # rate = 1e-2
    # accuracy = []
    # while(rate <= 1):
    #     cnt = 0
    #     correct = 0
    #     for it in x:
    #         l = after_pro(it, mean_girls, var_girls, 1 - rate)
    #         r = after_pro(it, mean_boys, var_boys, rate)
    #         res = 0 if(l > r) else 1
    #         if(res == y[cnt]):
    #             correct = correct + 1
    #         cnt = cnt + 1
    #     print('正确率　' + str(correct / len(x)))
    #     accuracy.append([rate, correct / len(x)])
    #     rate += 0.01
    # accuracy = numpy.array(accuracy)
    # plt.plot(accuracy[:, 0], accuracy[:, 1])
    # plt.title('正确率随男生先验概率的变化')
    # plt.xlabel('男生先验概率')
    # plt.ylabel('正确率')
    # plt.show()

    # x, y = getTest()
    # b = []
    # g = []
    # cnt = 0
    # for it in x:
    #     if(y[cnt] == 0):
    #         g.append(it)
    #     else:
    #         b.append(it)
    #     cnt = cnt + 1
    # b = numpy.array(b)
    # g = numpy.array(g)
    # rate = primary_components[1] / primary_components[0]
    # plt.scatter(g[:, 0], g[:, 1])
    # plt.scatter(b[:, 0], b[:, 1])
    # plt.plot([150, 195], [35, 35 + 45 * (rate)], color='red')
    # plt.legend(['新特征方向', '女生', '男生'])
    # plt.title('模式识别　1-Bayers 分类器')
    # plt.xlabel('男生先验概率')
    # plt.ylabel('正确率')
    # plt.show()
