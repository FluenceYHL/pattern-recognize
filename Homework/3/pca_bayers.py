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
        # print(cov_matrix)
        feature_values, feature_vectors = numpy.linalg.eig(
            numpy.mat(cov_matrix))  # 求特征值，特征向量
        # print(feature_values)
        # print(feature_values[1] / sum(feature_values))
        # print(feature_vectors)
        if(k == 0):
            k = self.__getK(feature_values)        # 如果没有指定　k, 就按照特征比率提取
        indexs = numpy.argsort(-feature_values)    # 对特征值从大到小排序，返回索引
        k_indexs = indexs[:k]                      # 提取　k 个特征值
        primary_components = feature_vectors[:, k_indexs]  # 找出对应的　k 个特征向量
        primary_components = abs(primary_components)
        # print(primary_components)
        return k, mean_value, primary_components, features.dot(primary_components)


def gaussi(x, mean_value, var_value):
    res = 1 / (math.sqrt(2 * math.pi * var_value))
    res = res * math.exp((-1 * (it - mean_value) *
                          (it - mean_value) / (2 * var_value)))
    return res


def after_pro(x, mean_value, var_value, pre):
    return gaussi(x, mean_value, var_value) * pre


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


if __name__ == '__main__':
    girls = getData('../dataSet/female.txt')
    boys = getData('../dataSet/male.txt')
    train_data = numpy.r_[girls, boys]
    y = numpy.r_[[0] * len(girls), [1] * len(boys)]

    # one = fisher()
    # one.fit(girls, boys)
    # x = numpy.r_[girls, boys]
    # y = [0] * len(girls)
    # for it in boys:
    #     y.append(1)
    # while(True):
    #     cnt = 0
    #     bad = []
    #     for it in x:
    #         res = one.predict(it)
    #         if(res != y[cnt]):
    #             bad.append(cnt)
    #         cnt = cnt + 1
    #     if(any(bad) == False):
    #         break
    #     print(bad)
    #     y = numpy.delete(y, bad, 0)
    #     res = []
    #     for it in y:
    #         if(it == 0):
    #             res.append(it)
    #     l = len(res)
    #     x = numpy.delete(x, bad, 0)
    #     girls = x[:l]
    #     boys = x[l:]
    #     one.fit(girls, boys)

    one = YHL_pca()
    k, mean_value, primary_components, new_x = one.pca(
        train_data, percentage=0.95, k=1)

    print(numpy.var(new_x))

    # new_girls = new_x[:len(girls)]
    # mean_girls = numpy.mean(new_girls)
    # var_girls = numpy.var(new_girls)

    # new_boys = new_x[len(girls):]
    # print(new_boys.shape)
    # mean_boys = numpy.mean(new_boys)
    # var_boys = numpy.var(new_boys)

    # x, y = getTest()
    # x = x - numpy.mean(x, axis=0)
    # x = x.dot(primary_components)
    # x.resize(x.shape[:1])
    # # print(x)
    # cnt = 0
    # correct = 0
    # for it in x:
    #     l = after_pro(it, mean_girls, var_girls, 0.1)
    #     r = after_pro(it, mean_boys, var_boys, 0.9)
    #     res = 0 if(l > r) else 1
    #     if(res == y[cnt]):
    #         correct = correct + 1
    #     cnt = cnt + 1
    # print('正确率　' + str(correct / len(x)))

    # check = 0
    # ex = x[:50]
    # ex.resize(50)
    # for it in ex:
    #     if(it >= -14):
    #         check += 1
    # print(check)

    # ex = x[50:]
    # ex.resize(250)
    # check = 0
    # for it in ex:
    #     if(it >= -14):
    #         check += 1
    # print(check)

    # cnt = 0
    # correct = 0
    # for it in x:
    #     if(it >= 0.94):
    #         res = 1
    #     else:
    #         res = 0
    #     if(res == y[cnt]):
    #         correct += 1
    #     cnt += 1
    # print(correct)

    # 搜索最佳分界点
    # lhs = numpy.min(new_x) - 1
    # rhs = numpy.max(new_x) + 1
    # print(lhs)
    # print(rhs)
    # sequence = numpy.meshgrid(numpy.arange(lhs, rhs, 1))[0]
    # # print(sequence)
    # # x, y = getTest()   # 注意这是在测试集...... 应该改成训练集
    # # x = x - numpy.mean(x, axis=0)
    # # x = x.dot(primary_components)
    # # x.resize(x.shape[:1])
    # x = numpy.r_[new_girls, new_boys]
    # print(x)
    # print(x)
    # ans = []
    # for border in sequence:
    #     # print(border)
    #     cnt = 0
    #     correct = 0
    #     for it in x:
    #         # print(it)
    #         if(it < border):
    #             res = 0
    #         else:
    #             res = 1
    #         if(res == y[cnt]):
    #             correct = correct + 1
    #         cnt = cnt + 1
    #     # print(str(border) + '  正确率  ' + str(correct / len(x)))
    #     ans.append([border, correct / len(x)])
    # ans = numpy.array(ans)
    # # print('最佳  :  ' + str(max(ans)))
    # plt.plot(ans[:, 0], ans[:, 1])
    # plt.show()

    # 正确率随着先验概率的变化
    # x, y = getTest()
    # x = x - numpy.mean(x, axis=0)
    # x = x.dot(primary_components)
    # x.resize(x.shape[:1])
    # print(x)
    # print(x.shape)
    # rate = 1e-2
    # accuracy = []
    # while(rate <= 1):
    #     cnt = 0
    #     correct = 0
    #     for it in x:
    #         l = after_pro(it, mean_girls, var_girls, rate)
    #         r = after_pro(it, mean_boys, var_boys, 1 - rate)
    #         res = 0 if(l > r) else 1
    #         if(res == y[cnt]):
    #             correct = correct + 1
    #         cnt = cnt + 1
    #     print('正确率　' + str(correct / len(x)))
    #     accuracy.append([1 - rate, correct / len(x)])
    #     rate += 0.01
    # accuracy = numpy.array(accuracy)
    # plt.plot(accuracy[:, 0], 1 - accuracy[:, 1])
    # plt.title('正确率随女生先验概率的变化')
    # plt.xlabel('女生先验概率')
    # plt.ylabel('错误率')
    # plt.show()

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
    # plt.scatter(g[:, 0], g[:, 1], color='red')
    # plt.scatter(b[:, 0], b[:, 1], color='green')
    # plt.legend(['女生', '男生'])
    # plt.show()

    # plt.hist(new_girls, 20, color='blue', alpha=0.8, rwidth=0.9)
    # girls_x = numpy.arange(-40, 40, 0.1)
    # girls_y = []
    # for it in girls_x:
    #     res = gaussi(it, mean_girls, var_girls)
    #     girls_y.append(res * 120)
    # plt.plot(girls_x, girls_y, color='green')
    # # plt.show()

    # plt.hist(new_boys, 20, color='black', alpha=0.8, rwidth=0.9)
    # boys_x = numpy.arange(-40, 40, 0.1)
    # boys_y = []
    # for it in boys_x:
    #     res = gaussi(it, mean_boys, var_boys)
    #     boys_y.append(res * 120)
    # plt.plot(boys_x, boys_y, color='red')
    # plt.legend(['女生密度分布', '男生密度分布'])
    # plt.show()

    # girls_x = numpy.arange(-40, 40, 0.1)
    # girls_y = []
    # for it in girls_x:
    #     2
    #     res = gaussi(it, mean_girls, var_girls)
    #     girls_y.append(res)
    # plt.plot(girls_x, girls_y)
    # # plt.show()

    # boys_x = numpy.arange(-40, 40, 0.1)
    # boys_y = []
    # for it in boys_x:
    #     res = gaussi(it, mean_boys, var_boys)
    #     boys_y.append(res)
    # plt.plot(boys_x, boys_y)
    # plt.legend(['女生类条件概率密度分布', '男生类条件概率密度分布'])
    # plt.show()

    # x, y = getTest()

    # x = numpy.r_[girls, boys]
    # y = [0] * len(girls)
    # for it in boys:
    #     y.append(1)
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
