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
    plt.figure(1)
    to_plot = {
        'tm': [],
        'pr': [],
        'th': [],
        'sv': [],
        'idx_pred': [],
        'idx_actual': [],
    }
    for xe, ye, ype in zip(data_x_test, data_y_test, data_y_pred):
        to_plot['tm'].append(xe[0])
        to_plot['pr'].append(xe[1])
        to_plot['th'].append(xe[2])
        to_plot['sv'].append(xe[3])
        to_plot['idx_pred'].append(ype)
        to_plot['idx_actual'].append(ye)

    print("Mean Absolute Error", mean_absolute_error(to_plot['idx_actual'], to_plot['idx_pred']))
    print("Mean Squared Error", mean_squared_error(to_plot['idx_actual'], to_plot['idx_pred']))
    # print(data_y_pred)
    # print(data_y_test)

    plt.subplot(2, 2, 1)
    plt.scatter(to_plot['tm'], to_plot['idx_actual'])
    plt.scatter(to_plot['tm'], to_plot['idx_pred'], color="red")
    # plt.yscale('linear')
    plt.title('TM')
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.scatter(to_plot['pr'], to_plot['idx_actual'])
    plt.scatter(to_plot['pr'], to_plot['idx_pred'], color="red")
    # plt.yscale('linear')
    plt.title('PR')
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.scatter(to_plot['th'], to_plot['idx_actual'])
    plt.scatter(to_plot['th'], to_plot['idx_pred'], color="red")
    # plt.yscale('linear')
    plt.title('TH')
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.scatter(to_plot['sv'], to_plot['idx_actual'])
    plt.scatter(to_plot['sv'], to_plot['idx_pred'], color="red")
    # plt.yscale('linear')
    plt.title('SV')
    plt.grid(True)

    plt.show()

    ####### Best fit ########

if __name__ == "__main__":
    main()
