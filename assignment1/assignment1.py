import matplotlib.pyplot as plt
import csv

from sklearn import linear_model


def main():
    filename = 'Dataset.csv'
    raw_data = open(filename, 'rt')
    reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
    x = list(reader)
    data_x_train, data_x_test, data_y_train, data_y_test = ([] for i in range(4))
    for train in x[20:]:
        data_y_train.append(float(train[4]))
        data_x_train.append((float(train[0]), float(train[1]), float(train[2]), float(train[3])))
    for test in x[1:20]:
        data_y_test.append(float(test[4]))
        data_x_test.append((float(test[0]), float(test[1]), float(test[2]), float(test[3])))
    # Create linear regression object
    regr = linear_model.LinearRegression()
    regr.fit(data_x_train, data_y_train)
    data_y_pred = regr.predict(data_x_test)
    plt.figure(1)
    for xe, ye in zip(data_x_test, data_y_test):
        plt.scatter(xe, [ye] * len(xe))
    plt.plot(data_x_test, data_y_pred)
    print(data_y_pred)
    print(data_y_test)
    plt.yscale('linear')
    plt.title('TM')
    plt.grid(True)
    plt.show()

    ####### Best fit ########
    plt.figure(2)

if __name__ == "__main__":
    main()
