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
        print('协方差矩阵')
        print(cov_matrix)
        feature_values, feature_vectors = numpy.linalg.eig(
            numpy.mat(cov_matrix))  # 求特征值，特征向量
        print('特征值')
        print(feature_values)
        print('特征向量')
        print(feature_vectors)
        if(k == 0):
            k = self.__getK(feature_values)        # 如果没有指定　k, 就按照特征比率提取
        indexs = numpy.argsort(-feature_values)    # 对特征值从大到小排序，返回索引
        k_indexs = indexs[:k]                      # 提取　k 个特征值
        primary_components = feature_vectors[:, k_indexs]  # 找出对应的　k 个特征向量
        primary_components = abs(primary_components)
        print('主成分')
        print(primary_components)
        return k, mean_value, primary_components, features.dot(primary_components)


# https://www.cnblogs.com/lzllovesyl/p/5235137.html
if __name__ == '__main__':
    girls = getData('../dataSet/female.txt')
    boys = getData('../dataSet/male.txt')
    train_data = numpy.r_[girls, boys]
    y = numpy.r_[[0] * len(girls), [1] * len(boys)]

    one = YHL_pca()
    k, mean_value, primary_components, new_x = one.pca(
        train_data, percentage=0.95, k=1)
    print(k)
    print(mean_value)
    print(new_x)
    new_x.resize(new_x.shape[:1])

    # one = PCA(n_components=1)
    # one.fit(train_data)
    # print(one.mean_)
    # print(one.singular_values_)
    # print(one.components_)
    # print(one.explained_variance_)
    # new_x = one.transform(train_data)
    # print(new_x)

    # b = []
    # g = []
    # cnt = 0
    # for it in new_x:
    #     if(y[cnt] == 0):
    #         g.append([it, 0])
    #     else:
    #         b.append([it, 0])
    #     cnt = cnt + 1
    # b = numpy.array(b)
    # g = numpy.array(g)

    # print(g.shape)
    # print(b.shape)

    # plt.scatter(g[:, 0], g[:, 1])
    # plt.scatter(b[:, 0], b[:, 1])
    # plt.legend(['女生', '男生'])
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

    print(g.shape)
    print(b.shape)
    rate = primary_components[1] / primary_components[0]
    plt.scatter(g[:, 0], g[:, 1])
    plt.scatter(b[:, 0], b[:, 1])
    plt.plot([150, 195], [35, 35 + 45 * (rate)], color='green')
    plt.legend(['新特征方向', '女生', '男生'])
    plt.title('模式识别　1-Bayers 分类器')
    plt.xlabel('男生先验概率')
    plt.ylabel('正确率')
    plt.show()
