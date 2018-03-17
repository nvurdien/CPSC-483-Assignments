import matplotlib.pyplot as plt
import csv

from sklearn import linear_model


def main():
    filename = 'Dataset.csv'
    raw_data = open(filename, 'rt')
    reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
    x = list(reader)
    X, y = ([] for _ in range(2))
    for test in x[1:]:
        y.append(float(test[4]))
        X.append((float(test[0]), float(test[1]), float(test[2]), float(test[3])))
    from sklearn.cross_validation import train_test_split
    data_x_train, data_x_test, data_y_train, data_y_test = train_test_split(X, y, test_size = 0.02, random_state = 0)
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

    ####### Polynomial order regression ########
    
    # Preprocessing: converting to polynomial matrix
    from sklearn.preprocessing import PolynomialFeatures
    poly_features = PolynomialFeatures(degree = 2)
    X_poly = poly_features.fit_transform(X)

    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size = 0.02, random_state = 0)

    # Fitting polynomial regression to the dataset
    from sklearn.linear_model import LinearRegression
    #poly_features.fit(X_poly, y_train)
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)

    # Model prediction using test set
    y_pred = lin_reg.predict(X_test)
    
    #plot results
    print(y_pred)
    print(data_y_test)

if __name__ == "__main__":
    main()
