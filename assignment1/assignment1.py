import matplotlib.pyplot as plt
import numpy as np
import csv

from matplotlib.ticker import NullFormatter
from sklearn import linear_model, preprocessing
from sklearn.metrics import mean_squared_error, r2_score


def main():
    filename = 'Dataset.csv'
    raw_data = open(filename, 'rt')
    reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
    x = list(reader)
    data_x_train, data_x_test, data_y_train, data_y_test = ([] for i in range(4))
    data_ytm_train, data_ytm_test, data_ypr_train, data_ypr_test, data_yth_train, data_yth_test, \
        data_ysv_train, data_ysv_test,  =  ([] for i in range(8))
    for train in x[20:]:
        data_x_train.append(float(train[4]))
        data_y_train.append((float(train[0]), float(train[1]), float(train[2]), float(train[3])))
        data_ytm_train.append(float(train[0]))
        data_ypr_train.append(float(train[1]))
        data_yth_train.append(float(train[2]))
        data_ysv_train.append(float(train[3]))

    for test in x[1:20]:
        data_x_test.append(float(test[4]))
        data_y_test.append((float(test[0]), float(test[1]), float(test[2]), float(test[3])))
        data_ytm_test.append(float(test[0]))
        data_ypr_test.append(float(test[1]))
        data_yth_test.append(float(test[2]))
        data_ysv_test.append(float(test[3]))

    # Create linear regression object
    regr = linear_model.LinearRegression()

    # Train the model using the training sets and reshape to fulfill 2D array requirement
    data_x_train = np.array(data_x_train).reshape((len(data_x_train), 1))
    # tm
    regr.fit(data_x_train, data_ytm_train)
    # Make predictions using the testing set and reshape to fulfill 2D array requirement
    data_x_test = np.array(data_x_test).reshape((len(data_x_test), 1))
    data_ytm_pred = regr.predict(data_x_test)
    # The coefficients
    print('Coefficients: \n', regr.coef_)
    # The mean squared error
    print("Mean squared error: %.2f" % mean_squared_error(data_ytm_test, data_ytm_pred))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(data_ytm_test, data_ytm_pred))
    # Plot outputs
    plt.figure(1)
    plt.scatter(data_x_test, data_ytm_test)
    plt.plot(data_x_test, data_ytm_pred, label="tm")
    plt.yscale('linear')
    plt.title('TM')
    plt.grid(True)

    # pr
    regr.fit(data_x_train, data_ypr_train)
    # Make predictions using the testing set and reshape to fulfill 2D array requirement
    data_ypr_pred = regr.predict(data_x_test)
    # The coefficients
    print('Coefficients: \n', regr.coef_)
    # The mean squared error
    print("Mean squared error: %.2f" % mean_squared_error(data_ytm_test, data_ytm_pred))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(data_ytm_test, data_ytm_pred))
    # Plot outputs
    plt.figure(2)
    plt.scatter(data_x_test, data_ypr_test)
    plt.plot(data_x_test, data_ypr_pred, label="pr")
    plt.yscale('linear')
    plt.title('PR')
    plt.grid(True)

    # th
    regr.fit(data_x_train, data_yth_train)
    # Make predictions using the testing set and reshape to fulfill 2D array requirement
    data_yth_pred = regr.predict(data_x_test)
    # The coefficients
    print('Coefficients: \n', regr.coef_)
    # The mean squared error
    print("Mean squared error: %.2f" % mean_squared_error(data_ytm_test, data_ytm_pred))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(data_ytm_test, data_ytm_pred))
    # Plot outputs
    plt.figure(3)
    plt.scatter(data_x_test, data_yth_test)
    plt.plot(data_x_test, data_yth_pred, label="th")
    plt.yscale('linear')
    plt.title('TH')
    plt.grid(True)

    # sv
    regr.fit(data_x_train, data_ysv_train)
    # Make predictions using the testing set and reshape to fulfill 2D array requirement
    data_ysv_pred = regr.predict(data_x_test)
    # The coefficients
    print('Coefficients: \n', regr.coef_)
    # The mean squared error
    print("Mean squared error: %.2f" % mean_squared_error(data_ytm_test, data_ytm_pred))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(data_ytm_test, data_ytm_pred))
    # Plot outputs
    plt.figure(4)
    plt.scatter(data_x_test, data_ysv_test)
    plt.plot(data_x_test, data_ysv_pred, label="sv")
    plt.yscale('linear')
    plt.title('SV')
    plt.grid(True)

    # Format the minor tick labels of the y-axis into empty strings with
    # `NullFormatter`, to avoid cumbering the axis with too many labels.
    plt.gca().yaxis.set_minor_formatter(NullFormatter())
    # Adjust the subplot layout, because the logit one may take more space
    # than usual, due to y-tick labels like "1 - 10^{-3}"
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                        wspace=0.35)

    plt.show()


if __name__ == "__main__":
    main()
