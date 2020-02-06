import csv
import numpy
import random

Nin = 0
Nhid = 0
Nout = 0
epochs = 0
eta = 0.0
alfa = 0.0
bias = 0


def f(x):
    return numpy.tanh(x)


def fprim(x):
    return 1 - numpy.tanh(x)*numpy.tanh(x)


def read_config_data(filename):
    file = open(filename, 'rt')
    config = csv.reader(file, delimiter=",")

    next(config)

    global Nin, Nhid, Nout, epochs, eta, alfa, bias

    for line in config:
        Nin = int(line[0])
        Nhid = int(line[1])
        Nout = int(line[2])
        epochs = int(line[3])
        eta = float(line[4])
        alfa = float(line[5])
        bias = float(line[6])

    file.close()


def read_input_data(filename):
    X = []
    Y = []

    file = open(filename, "r")
    data = csv.reader(file, delimiter=",", quoting=csv.QUOTE_NONNUMERIC)

    for row in data:
        X.append(row[0:Nin])
        Y.append(row[Nin])

    file.close()

    return X, Y


def initialize_weights():
    w1 = []
    w2 = []

    old_w1 = []
    old_w2 = []

    for i in range(Nhid):
        w = []
        old = []

        for j in range(Nin):
            w.append(random.random()/10 - 0.05)
            old.append(0.0)
        w1.append(w)
        old_w1.append(old)

    for i in range(Nout):
        w = []
        old = []

        for j in range(Nhid):
            w.append(random.random()/10 - 0.05)
            old.append(0.0)
        w2.append(w)
        old_w2.append(old)

    return w1, w2, old_w1, old_w2


def train(w1, w2, old_w1, old_w2, X, Y):
    for epoch in range(epochs):
        error = 0.0

        for p in range(len(X)):
            v0 = []
            v1 = []
            v2 = []

            d1 = []
            d2 = []

            z1 = []
            z2 = []

            sum1 = 0.0
            sum2 = 0.0

            # przypisanie wektora treningowexo X do V0
            for i in range(Nin):
                v0.append(X[p][i])

            # wartosci dla neuronow wartwy ukrytej v1
            for i in range(Nhid):
                sum1 = 0.0
                for j in range(Nin):
                    sum1 += w1[i][j]*v0[j]
                v1.append(f(sum1))

            # wartosci dla neuronow wartwy wyjsciowej v2
            for i in range(Nout):
                sum2 = 0.0
                for j in range(Nhid):
                    sum2 += w2[i][j]*v1[j]
                v2.append(f(sum2))

            # blad na wyjsciu wartwy wyjsciowej
            for i in range(Nout):
                d2.append((fprim(sum2) + bias) * (Y[p] - v2[i]))

            # blad na wyjsciu wartwy ukrytej
            for i in range(Nhid):
                sum3 = 0.0
                for j in range(Nout):
                    sum3 += w2[j][i] * d2[j]
                d1.append(fprim(sum1) * sum3)

            # zmiana wag dla warswty wyjsciowej
            for i in range(Nout):
                z = []
                for j in range(Nhid):
                    z.append(eta * d2[i] * v1[j])
                    w2[i][j] += z[j] + alfa * old_w2[i][j]
                z2.append(z)

            # zmiana wag dla warstwy ukrytej
            for i in range(Nhid):
                z = []
                for j in range(Nin):
                    z.append(eta * d1[i] * v0[j])
                    w1[i][j] += z[j] + alfa * old_w1[i][j]
                z1.append(z)

            # blad sredni kwadratowy
            for i in range(Nout):
                error += ((Y[p]-v2[i])**2)/2.0

            # aktualizacja starych wag
            for i in range(Nhid):
                for j in range(Nin):
                    old_w1[i][j] = z1[i][j]

            for i in range(Nout):
                for j in range(Nhid):
                    old_w2[i][j] = z2[i][j]

        if epoch % 100 == 0:
            print("Epoch = ", epoch, ". Error = ", error)

    return w1, w2


def test(filename, w1, w2):
    Xtest, Yexpected = read_input_data(filename)
    Yout = []

    for p in range(len(Xtest)):
        v1 = []

        for i in range(Nhid):
            sum1 = 0.0
            for j in range(Nin):
                sum1 += w1[i][j] * Xtest[p][j]
            v1.append(f(sum1))

        for i in range(Nout):
            sum2 = 0.0
            for j in range(Nhid):
                sum2 += w2[i][j] * v1[j]
            Yout.append(f(sum2))

    return Yout, Yexpected


if __name__ == '__main__':
    read_config_data("data/config.csv")
    Xtrain, Ytrain = read_input_data("data/train_data.csv")

    w1, w2, old_w1, old_w2 = initialize_weights()

    wt1, wt2 = train(w1, w2, old_w1, old_w2, Xtrain, Ytrain)

    print("WAGI:")
    print(wt1)
    print(wt2)

    Yout, Yexpected = test("data/test_data.csv", wt1, wt2)

    for i in range(len(Yout)):
        print("Results:", Yout[i])
        print("Expected results:", Yexpected[i])
