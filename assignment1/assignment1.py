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
    
    # Fitting polynomial regression to the dataset
    from sklearn.preprocessing import PolynomialFeatures
    poly_reg = PolynomialFeatures(degree = 3)
    X_poly = poly_reg.fit_transform(data_x_train) #getting polynomial matrix
    poly_reg.fit(X_poly, data_y_train)
    #X_poly = X_poly[:, 1:5]
    lin_reg = linear_model.LinearRegression()
    lin_reg.fit(X_poly, data_y_train) #fitting linear regression to polynomial matrix

    # Model prediction using test set
    y_pred = lin_reg.predict(data_x_test)
    print(y_pred)
    #print(data_y_test)
    print (data_y_pred)
    
    #plot results
    plt.figure(2)

if __name__ == "__main__":
    main()
