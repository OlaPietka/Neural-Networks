import numpy
import csv
import random

Nin = 0
Nhid = 0
Nout = 0
epochs = 0
eta = 0.0


def g(x):
    return numpy.tanh(x)


def gprim(x):
    return 1 - numpy.tanh(x) * numpy.tanh(x)


def read_config_data(filename):
    file = open(filename, 'rt')
    config = csv.reader(file, delimiter=",")

    next(config)

    global Nin, Nhid, Nout, epochs, eta

    for line in config:
        Nin = int(line[0])
        Nhid = int(line[1])
        Nout = int(line[2])
        epochs = int(line[3])
        eta = float(line[4])
    file.close()


def read_input_data(filename):
    X = []
    Y = []

    file = open(filename, 'rt')
    file.seek(0)

    dataset = csv.reader(file, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)

    for line in dataset:
        X.append(line[0:Nin])
        Y.append(line[Nin])
    file.close()

    return X, Y


def initialize_weights():
    w1 = []
    w2 = []

    for i in range(Nhid):
        w = []
        for j in range(Nin):
            w.append(random.random()/10.0 - 0.05)
        w1.append(w)

    for i in range(Nout):
        w = []
        for j in range(Nhid):
            w.append(random.random()/ 10.0 - 0.05)
        w2.append(w)

    return w1, w2


def train(X, Y, w1, w2):
    for epoch in range(epochs):
        error = 0.0

        for p in range(len(X)):
            K = []
            M = []

            d1 = []
            d2 = []

            sum1 = 0.0
            sum2 = 0.0

            # wartosci dla neuronow wartwy ukrytej K
            for i in range(Nhid):
                sum1 = 0.0
                for j in range(Nin):
                    sum1 += w1[i][j] * X[p][j]
                K.append(g(sum1))

            # wartosci dla neuronow wartwy wyjsciowej M
            for i in range(Nout):
                sum2 = 0.0
                for j in range(Nhid):
                    sum2 += w2[i][j]*K[j]
                M.append(g(sum2))

            # blad na wyjsciu wartwy wyjsciowej
            for i in range(Nout):
                d2.append(gprim(sum2)*(Y[p] - M[i]))

            # blad na wyjsciu wartwy ukrytej
            for i in range(Nhid):
                sum3 = 0.0
                for j in range(Nout):
                    sum3 += w2[j][i]*d2[j]
                d1.append(gprim(sum1)*sum3)

            # zmiana wag dla warswty wyjsciowej
            for i in range(Nout):
                for j in range(Nhid):
                    w2[i][j] += eta * d2[i] * K[j]

            # zmiana wag dla warstwy ukrytej
            for i in range(Nhid):
                for j in range(Nin):
                    w1[i][j] += eta * d1[i] * X[p][j]

            # blad sredni kwadratowy
            for i in range(Nout):
                error += ((Y[p] - M[i]) ** 2) / 2.0

        if epoch % 100 == 0:
            print("Epoch = ", epoch, ". Error = ", error)

    return w1, w2


def test(filename, w1, w2):
    Xtest, Yexpected = read_input_data(filename)
    Yout = []

    for p in range(len(Xtest)):
        K = []

        for i in range(Nhid):
            sum1 = 0.0
            for j in range(Nin):
                sum1 += w1[i][j] * Xtest[p][j]
            K.append(g(sum1))

        for i in range(Nout):
            sum2 = 0.0
            for j in range(Nhid):
                sum2 += w2[i][j] * K[j]
            Yout.append(g(sum2))

    return Yout, Yexpected


if __name__ == '__main__':
    read_config_data("data/config.csv")

    Xtrain, Ytrain = read_input_data("data/train_data.csv")

    w1, w2 = initialize_weights()

    w1, w2 = train(Xtrain, Ytrain, w1, w2)

    print("WAGI:")
    print(w1)
    print(w2)

    Yout, Yexpected = test("data/test_data.csv", w1, w2)
    print("Results:", Yout)
    print("Expected results:", Yexpected)
