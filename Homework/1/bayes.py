# package
import math
import numpy
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

    l = len(girls)
    r = len(boys)
    pre_girls = l / (l + r)
    pre_boys = r / (l + r)

    x, y = getTest()
    cnt = 0
    correct = 0
    for it in x:
        lhs = -0.5 * ((it - mean_girls).T.dot(inv_girls).dot(it -
                                                             mean_girls) + math.log(det_girls)) + math.log(pre_girls)
        rhs = -0.5 * ((it - mean_boys).T.dot(inv_boys).dot(it -
                                                           mean_boys) + math.log(det_boys)) + math.log(pre_boys)
        res = 0 if(lhs > rhs) else 1
        if(res == y[cnt]):
            correct = correct + 1
        cnt = cnt + 1
    print(correct / len(x))
