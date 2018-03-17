import matplotlib.pyplot as plt
import csv
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error, mean_squared_error


def main():
    filename = 'Dataset.csv'
    raw_data = open(filename, 'rt')
    reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
    x = list(reader)
    x = x[1:]
    data_x_train, data_x_test, data_y_train, data_y_test = ([] for _ in range(4))
    for train in x[20:]:
        data_y_train.append(float(train[4]))
        data_x_train.append((float(train[0]), float(train[1]), float(train[2]), float(train[3])))
    for test in x[:200]:
        data_y_test.append(float(test[4]))
        data_x_test.append((float(test[0]), float(test[1]), float(test[2]), float(test[3])))
    # Create linear regression object
    regr = linear_model.LinearRegression()
    regr.fit(data_x_train, data_y_train)
    data_y_pred = regr.predict(data_x_test)
    print("Mean Absolute Error", mean_absolute_error(data_y_test, data_y_pred))
    print("Mean Squared Error", mean_squared_error(data_y_test, data_y_pred))

    ####### Best fit ########

if __name__ == "__main__":
    main()
