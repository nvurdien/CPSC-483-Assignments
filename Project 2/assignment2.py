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
    OVER_50 = 0, '>50K'
    LESS_THAN_50 = 1, '<=50K'
    OTHER = 2, 'Unknown'

    def __new__(cls, value, name):
        member = object.__new__(cls)
        member._value_ = value
        member.fullname = name
        return member

    def __int__(self):
        return self.value


def main():
    filename = 'Dataset.csv'
    raw_data = open(filename, 'rt')
    reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
    x = list(reader)

    label_names = ['<=50K', '>50K']
    labels = []
    feature_names = ['age', 'work_class', 'education', 'years_education', 'marital_status', 'occupation',
                     'relationship', 'gender', 'capital_gain', 'capital_loss', 'hours']
    features = []

    for train in x:
        income, work_class = ''
        for name, member in Income.__members__.items():
            if train[11].strip() == member.fullname:
                income = member.value
        if income == '':
            labels.append(Income.OTHER)
        else:
            labels.append(income)

        for name, member in WorkClass.__members__.items():
            if train[1].strip() == member.fullname:
                work_class = member.value

        if train[1].strip() == 'Private':
            work_class = WorkClass.PRIVATE
        elif train[1].strip() == 'Self-emp-not-inc':
            work_class = WorkClass.SELF_EMP_NOT_INC
        elif train[1].strip() == 'Self-emp-inc':
            work_class = WorkClass.SELF_EMP_INC
        elif train[1].strip() == 'Federal-gov':
            work_class = WorkClass.FEDERAL_GOV
        elif train[1].strip() == 'Local-gov':
            work_class = WorkClass.LOCAL_GOV
        elif train[1].strip() == 'State-gov':
            work_class = WorkClass.STATE_GOV
        elif train[1].strip() == 'Without-pay':
            work_class = WorkClass.WITHOUT_PAY
        elif train[1].strip() == 'Never-worked':
            work_class = WorkClass.NEVER_WORKED
        else:
            work_class = WorkClass.OTHER

        if train[2].strip() == 'Bachelors':
            education = EducationLevel.BACHELORS
        elif train[2].strip() == 'Some-college':
            education = EducationLevel.SOME_COLLEGE
        elif train[2].strip() == '11th':
            education = EducationLevel.ELEVENTH
        elif train[2].strip() == 'HS-grad':
            education = EducationLevel.HS_GRAD
        elif train[2].strip() == 'Prof-school':
            education = EducationLevel.PROF_SCHOOL
        elif train[2].strip() == 'Assoc-acdm':
            education = EducationLevel.ASSOC_ACDM
        elif train[2].strip() == 'Assoc-voc':
            education = EducationLevel.ASSOC_VOC
        elif train[2].strip() == '9th':
            education = EducationLevel.NINTH
        elif train[2].strip() == '7th-8th':
            education = EducationLevel.SEVENTH_EIGTH
        elif train[2].strip() == '12th':
            education = EducationLevel.TWELVETH
        elif train[2].strip() == 'Masters':
            education = EducationLevel.MASTERS
        elif train[2].strip() == '1st-4th':
            education = EducationLevel.FIRST_FOURTH
        elif train[2].strip() == '10th':
            education = EducationLevel.TENTH
        elif train[2].strip() == 'Doctorate':
            education = EducationLevel.DOCTORATE
        elif train[2].strip() == '5th-6th':
            education = EducationLevel.FIFTH_SIXTH
        elif train[2].strip() == 'Preschool':
            education = EducationLevel.PRESCHOOL
        else:
            education = EducationLevel.OTHER

        if train[4].strip() == 'Married-civ-spouse':
            marital_status = MaritalStatus.MARRIED_CIV_SPOUSE
        elif train[4].strip() == 'Divorced':
            marital_status = MaritalStatus.DIVORCED
        elif train[4].strip() == 'Never-married':
            marital_status = MaritalStatus.NEVER_MARRIED
        elif train[4].strip() == 'Separated':
            marital_status = MaritalStatus.SEPARATED
        elif train[4].strip() == 'Widowed':
            marital_status = MaritalStatus.WIDOWED
        elif train[4].strip() == 'Married-spouse-absent':
            marital_status = MaritalStatus.MARRIED_SPOUSE_ABSENT
        elif train[4].strip() == 'Married-AF-spouse':
            marital_status = MaritalStatus.MARRIED_AF_SPOUSE
        else:
            marital_status = MaritalStatus.OTHER

        if train[5].strip() == 'Tech-support':
            occupation = Occupation.TECH_SUPPORT
        elif train[5].strip() == 'Craft-repair':
            occupation = Occupation.CRAFT_REPAIR
        elif train[5].strip() == 'Other-service':
            occupation = Occupation.OTHER_SERVICE
        elif train[5].strip() == 'Sales':
            occupation = Occupation.SALES
        elif train[5].strip() == 'Exec-managerial':
            occupation = Occupation.EXEC_MANAGERIAL
        elif train[5].strip() == 'Transport-moving':
            occupation = Occupation.TRANSPORT_MOVING
        elif train[5].strip() == 'Priv-house-serv':
            occupation = Occupation.PRIV_HOUSE_SERV
        elif train[5].strip() == 'Protective-serv':
            occupation = Occupation.PROTECTIVE_SERV
        elif train[5].strip() == 'Armed-Forces':
            occupation = Occupation.ARMED_FORCES
        else:
            occupation = Occupation.OTHER

        if train[6].strip() == 'Wife':
            relationship = Relationship.WIFE
        elif train[6].strip() == 'Own-child':
            relationship = Relationship.OWN_CHILD
        elif train[6].strip() == 'Husband':
            relationship = Relationship.HUSBAND
        elif train[6].strip() == 'Not-in-family':
            relationship = Relationship.NOT_IN_FAMILY
        elif train[6].strip() == 'Other-relative':
            relationship = Relationship.OTHER_RELATIVE
        elif train[6].strip() == 'Unmarried':
            relationship = Relationship.UNMARRIED
        else:
            relationship = Relationship.OTHER

        if train[7].strip() == 'Female':
            gender = Gender.FEMALE
        elif train[7].strip() == 'Male':
            gender = Gender.MALE
        else:
            gender = Gender.OTHER

        features.append([float(train[0].strip()), work_class.value, education.value, float(train[3].strip()), marital_status.value,
                         occupation.value, relationship.value, gender.value, float(train[8].strip()), float(train[9].strip()),
                         float(train[10].strip())])

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
