import csv


def main():
    filename = 'Data.csv'
    raw_data = open(filename, 'rt')
    reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
    x = list(reader)


if __name__ == "__main__":
    main()
