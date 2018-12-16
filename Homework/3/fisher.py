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
    return collect


if __name__ == '__main__':
    girls = getData('../dataSet/female.txt')
    boys = getData('../dataSet/male.txt')
    train_data = numpy.r_[girls, boys]
    y = numpy.r_[[0] * len(girls), [1] * len(boys)]

    mean_girls = numpy.mean(girls, axis=0)
    cov_girls = numpy.dot((girls - mean_girls).T, girls - mean_girls)

    mean_boys = numpy.mean(boys, axis=0)
    cov_boys = numpy.dot((boys - mean_boys).T, boys - mean_boys)

    Sw = cov_girls = cov_boys
    inv_Sw = numpy.linalg.inv(Sw)
    w = inv_Sw.dot(mean_girls - mean_boys)
    print(w)

    w0 = -0.5 * (mean_girls + mean_boys).dot(inv_Sw).dot(mean_girls -
                                                         mean_boys) - math.log(len(mean_boys) / len(mean_girls))

    print('w0 = ' + str(w0))
    x, y = getTest()
    cnt = 0
    correct = 0
    for it in x:
        res = w.dot(it) + w0
        res = 0 if(res > 0) else 1
        if(res == y[cnt]):
            correct += 1
        cnt = cnt + 1
    print('正确率  ' + str(correct / len(x)))
