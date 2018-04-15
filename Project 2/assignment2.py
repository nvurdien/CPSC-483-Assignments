import csv
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB


def main():
    filename = 'Dataset.csv'
    raw_data = open(filename, 'rt')
    reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
    x = list(reader)
    X, y = ([] for _ in range(2))
    for train in x:
        if train[8] == '<=50K':
            y.append(1)
        else:
            y.append(0)
        X.append((train[0], train[1], train[2], train[3], train[4], train[5],
                 train[6], train[7]))
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X)
    clf = MultinomialNB().fit(X_train_tfidf, y)


if __name__ == "__main__":
    main()
