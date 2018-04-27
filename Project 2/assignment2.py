import csv
from enum import Enum
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.neural_network import MLPClassifier


class WorkClass(Enum):
    PRIVATE = 0, 'Private'
    SELF_EMP_NOT_INC = 1, 'Self-emp-not-inc'
    SELF_EMP_INC = 2, 'Self-emp-inc'
    FEDERAL_GOV = 3, 'Federal-gov'
    LOCAL_GOV = 4, 'Local-gov'
    STATE_GOV = 5, 'State-gov'
    WITHOUT_PAY = 6, 'Without-pay'
    NEVER_WORKED = 7, 'Never-worked'
    OTHER = 8, 'Unknown'

    def __new__(cls, value, name):
        member = object.__new__(cls)
        member._value_ = value
        member.fullname = name
        return member

    def __int__(self):
        return self.value


class EducationLevel(Enum):
    BACHELORS = 0, 'Bachelors'
    SOME_COLLEGE = 1, 'Some-college'
    ELEVENTH = 2, '11th'
    HS_GRAD = 3, 'HS-grad'
    PROF_SCHOOL = 4, 'Prof-school'
    ASSOC_ACDM = 5, 'Assoc-acdm'
    ASSOC_VOC = 6, 'Assoc-voc'
    NINTH = 7, '9th'
    SEVENTH_EIGTH = 8, '7th-8th'
    TWELVETH = 9, '12th'
    MASTERS = 10, 'Masters'
    FIRST_FOURTH = 11, '1st-4th'
    TENTH = 12, '10th'
    DOCTORATE = 13, 'Doctorate'
    FIFTH_SIXTH = 14, '5th-6th'
    PRESCHOOL = 15, 'Preschool'
    OTHER = 16, 'Unknown'

    def __new__(cls, value, name):
        member = object.__new__(cls)
        member._value_ = value
        member.fullname = name
        return member

    def __int__(self):
        return self.value


class MaritalStatus(Enum):
    MARRIED_CIV_SPOUSE = 0, 'Married-civ-spouse'
    DIVORCED = 1, 'Divorced'
    NEVER_MARRIED = 2, 'Never-married'
    SEPARATED = 3, 'Separated'
    WIDOWED = 4, 'Widowed'
    MARRIED_SPOUSE_ABSENT = 5, 'Married-spouse-absent'
    MARRIED_AF_SPOUSE = 6, 'Married-AF-spouse'
    OTHER = 7, 'Unknown'

    def __new__(cls, value, name):
        member = object.__new__(cls)
        member._value_ = value
        member.fullname = name
        return member

    def __int__(self):
        return self.value


class Occupation(Enum):
    TECH_SUPPORT = 0, 'Tech-support'
    CRAFT_REPAIR = 1, 'Craft-repair'
    OTHER_SERVICE = 2, 'Other-service'
    SALES = 3, 'Sales'
    EXEC_MANAGERIAL = 4, 'Exec-managerial'
    PROF_SPECIALTY = 5, 'Prof-specialty'
    HANDLERS_CLEANERS = 6, 'Handlers-cleaners'
    MACHINE_OP_INSPCT = 7, 'Machine-op-inspct'
    ADM_CLERICAL = 8, 'Adm-clerical'
    FARMING_FISHING = 9, 'Farming-fishing'
    TRANSPORT_MOVING = 10, 'Transport-moving'
    PRIV_HOUSE_SERV = 11, 'Priv-house-serv'
    PROTECTIVE_SERV = 12, 'Protective-serv'
    ARMED_FORCES = 13, 'Armed-Forces'
    OTHER = 14, 'Unknown'

    def __new__(cls, value, name):
        member = object.__new__(cls)
        member._value_ = value
        member.fullname = name
        return member

    def __int__(self):
        return self.value


class Relationship(Enum):
    WIFE = 0, 'Wife'
    OWN_CHILD = 1, 'Own-child'
    HUSBAND = 2, 'Husband'
    NOT_IN_FAMILY = 3, 'Not-in-family'
    OTHER_RELATIVE = 4, 'Other-relative'
    UNMARRIED = 5, 'Unmarried'
    OTHER = 6, 'Unknown'

    def __new__(cls, value, name):
        member = object.__new__(cls)
        member._value_ = value
        member.fullname = name
        return member

    def __int__(self):
        return self.value


class Gender(Enum):
    FEMALE = 0, 'Female'
    MALE = 1, 'Male'
    OTHER = 2, 'Unknown'

    def __new__(cls, value, name):
        member = object.__new__(cls)
        member._value_ = value
        member.fullname = name
        return member

    def __int__(self):
        return self.value


class Income(Enum):
    LESS_THAN_50 = 0, '<=50K'
    OVER_50 = 1, '>50K'
    OTHER = 2, 'Unknown'

    def __new__(cls, value, name):
        member = object.__new__(cls)
        member._value_ = value
        member.fullname = name
        return member

    def __int__(self):
        return self.value


# compares an value to see if it matches the given state
# returns matching state or other if no matching state is found
def comparison(value, state):
    for name, member in state.__members__.items():
        if value.strip() == member.fullname:
            return member
    return state.OTHER


def main():
    # creates list from csv file
    filename = 'Dataset.csv'
    raw_data = open(filename, 'rt')
    reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
    x = list(reader)

    # instantiate necessary variables for classification
    # labels - what we are trying to find
    labels = []
    # feature_names - categories available
    feature_names = [
        'age', 'work_class', 'education', 'years_education', 'marital_status', 'occupation', 'relationship', 'gender',
        'capital_gain', 'capital_loss', 'hours'
    ]
    # features - feature values
    features = []
    # features_w_names - names of each feature
    features_w_names = []

    for train in x:
        # finds the enumeration associated with each value in work class, education, marital status, occupation,
        # relationship and gender
        work_class = comparison(train[1], WorkClass)
        education = comparison(train[2], EducationLevel)
        marital_status = comparison(train[4], MaritalStatus)
        occupation = comparison(train[5], Occupation)
        relationship = comparison(train[6], Relationship)
        gender = comparison(train[7], Gender)

        # adds income value to the labels array because thats what we want to find
        labels.append(comparison(train[11], Income).value)

        # appends rest of values into available features
        features.append([
            float(train[0].strip()), work_class.value, education.value, float(train[3].strip()),
            marital_status.value, occupation.value, relationship.value, gender.value,
            float(train[8].strip()), float(train[9].strip()), float(train[10].strip())
        ])

        # appends feature names into a separate array
        features_w_names.append([
            float(train[0].strip()), work_class.fullname, education.fullname,
            float(train[3].strip()), marital_status.fullname, occupation.fullname,
            relationship.fullname, gender.fullname, float(train[8].strip()),
            float(train[9].strip()), float(train[10].strip())
        ])

    # sets up training and testing arrays for classification
    train, test, train_labels, test_labels = train_test_split(features, labels, test_size=0.2, random_state=0)

    # Naive Bayes #

    naiveBayes = GaussianNB()
    naiveBayesModel = naiveBayes.fit(train, train_labels)
    naiveBayesPredicted = naiveBayes.predict(test)
    print(naiveBayesPredicted)
    print(accuracy_score(test_labels, naiveBayesPredicted))

    # Decision Tree #

    decisionTree = tree.DecisionTreeClassifier()
    decisionTreeModel = decisionTree.fit(train, train_labels)
    decisionTreePredicted = decisionTree.predict(test)
    print(decisionTreePredicted)
    print(accuracy_score(test_labels, decisionTreePredicted))

    # Multilayer Perceptron #

    multilayerPerceptron = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=0)
    multilayerPerceptronModel = multilayerPerceptron.fit(train, train_labels)
    multilayerPerceptronPredicted = multilayerPerceptron.predict(test)
    print(multilayerPerceptronPredicted)
    print(accuracy_score(test_labels, multilayerPerceptronPredicted))


if __name__ == "__main__":
    main()
