import csv
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
# import matplotlib.pyplot as plt


def main():
    filename = 'Dataset.csv'
    raw_data = open(filename, 'rt')
    reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
    x = list(reader)
    x = x[1:]
    X, y = ([] for _ in range(2))
    for train in x[1:]:
        y.append(float(train[4]))
        X.append((float(train[0]), float(train[1]), float(train[2]), float(train[3])))

    data_x_train, data_x_test, data_y_train, data_y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    ######################## Multiple Linear Regression ########################

    regr = linear_model.LinearRegression()
    regr.fit(data_x_train, data_y_train)
    data_y_pred = regr.predict(data_x_test)
    print("\nMultiple Linear Regression Errors:")
    print("Mean Absolute Error", mean_absolute_error(data_y_test, data_y_pred))
    print("Mean Squared Error", mean_squared_error(data_y_test, data_y_pred))

    ######################## Polynomial fit ########################

    # Preprocessing: converting to polynomial matrix

    poly_features = PolynomialFeatures(degree=5)
    X_poly = poly_features.fit_transform(X)

    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.02, random_state=0)

    # Fitting polynomial regression to the dataset
    from sklearn.linear_model import LinearRegression
    # poly_features.fit(X_poly, y_train)
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)

    # Model prediction using test set
    y_pred = lin_reg.predict(X_test)

    # plot results
    # print(y_pred)
    # print(data_y_test)
    print("\nPolynomial Fit Errors:")
    print("Mean Absolute Error", mean_absolute_error(y_test, y_pred))
    print("Mean Squared Error", mean_squared_error(y_test, y_pred), "\n")


if __name__ == "__main__":
    main()
