# python 3.7
# Scikit-learn ver. 0.23.2
from sklearn.linear_model import LogisticRegression, SGDClassifier
from scipy.special import expit
# matplotlib 3.3.1
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report, plot_confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer, LabelEncoder, scale
from sklearn.model_selection import train_test_split


"""
TO DO:
 - try to run predictNew with the same data and see if the accuracy is the same
 - test different classifiers and scalers
 - Play with vars on MLPClassifier
"""


def main():
    data = pd.read_csv('adult.csv', sep=',')

    #Preprocessing
    data = data.head(1000)   #Only consider the first few entries to speed it up
    data = data[data['workclass'] != "?"]
    data = data[data['occupation'] != "?"]
    data = data[data['native-country'] != "?"]
    data['workclass'] = data['workclass'].astype('str')

    # print("1st 5 Rows of Data: \n" + str(data.head()) + "\n")  #Prints top 5 rows of the table
    print("Feature Datatype Info: \n" + str(data.info()) + "\n")  #Prints info about each column's datatype, etc
    # print("Number of Null Values Per Feature: \n" + str(data.isnull().sum()) + "\n")  #Prints how many null values there are for each column
    print(str(data['income'].value_counts()) + "\n")    #Prints a table with just the data number and the corresponding income value

    #Converting String Values to Integer
    global workclassStrIntCorrespondence
    global educationStrIntCorrespondence
    global maritalStatusStrIntCorrespondence
    global occupationStrIntCorrespondence
    global relationshipStrIntCorrespondence
    global raceStrIntCorrespondence
    global genderStrIntCorrespondence
    global nativeCountryStrIntCorrespondence
    global incomeStrIntCorrespondence
    #Possible values were found in the Preprecessing program and organized manually
    workclassStrIntCorrespondence = ['Never-worked', 'Without-pay', 'Local-gov', 'State-gov', 'Federal-gov', 'Private', 'Self-emp-not-inc', 'Self-emp-inc']
    educationStrIntCorrespondence = ['Preschool', '1st-4th', '5th-6th', '7th-8th', '9th', '10th', '11th', '12th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', 'Some-college', 'Bachelors', 'Masters', 'Doctorate']
    maritalStatusStrIntCorrespondence = ['Never-married', 'Divorced', 'Separated', 'Married-AF-spouse', 'Married-civ-spouse',  'Widowed', 'Married-spouse-absent']
    occupationStrIntCorrespondence = [ 'Handlers-cleaners', 'Armed-Forces', 'Priv-house-serv', 'Farming-fishing', 'Protective-serv', 'Tech-support',  'Craft-repair', 'Sales', 'Adm-clerical', 'Machine-op-inspct', 'Other-service', 'Transport-moving', 'Prof-specialty', 'Exec-managerial']
    relationshipStrIntCorrespondence = ['Not-in-family', 'Other-relative' 'Unmarried', 'Own-child', 'Husband', 'Wife', ]
    raceStrIntCorrespondence = ['Black',  'Amer-Indian-Eskimo', 'Other', 'Asian-Pac-Islander', 'White']
    genderStrIntCorrespondence = ['Female', 'Male']
    nativeCountryStrIntCorrespondence = [ 'Trinadad&Tobago', 'Haiti', 'India', 'Nicaragua', 'Honduras', 'Jamaica', 'Ecuador', 'Yugoslavia', 'Columbia', 'Cambodia', 'Peru', 'Guatemala', 'Mexico', 'Dominican-Republic', 'Philippines', 'Thailand', 'El-Salvador', 'Puerto-Rico', 'Vietnam', 'South', 'Outlying-US(Guam-USVI-etc)', 'Columbia', 'Japan', 'Poland', 'Laos', 'England', 'Cuba', 'Taiwan', 'Italy', 'Canada', 'Portugal', 'China', 'Iran', 'Scotland', 'Hungary', 'Hong', 'Greece', 'France', 'Ireland', 'Germany', 'Holand-Netherlands', 'United-States']
    incomeStrIntCorrespondence = ['<=50K', '>50K']
    data['workclass'] = convertStrInt(data['workclass'], workclassStrIntCorrespondence)
    data['education'] = convertStrInt(data['education'], educationStrIntCorrespondence)
    data['marital-status'] = convertStrInt(data['marital-status'], maritalStatusStrIntCorrespondence)
    data['occupation'] = convertStrInt(data['occupation'], occupationStrIntCorrespondence)
    data['relationship'] = convertStrInt(data['relationship'], relationshipStrIntCorrespondence)
    data['race'] = convertStrInt(data['race'], raceStrIntCorrespondence)
    data['gender'] = convertStrInt(data['gender'], genderStrIntCorrespondence)
    data['native-country'] = convertStrInt(data['native-country'], nativeCountryStrIntCorrespondence)
    data['income'] = convertStrInt(data['income'], incomeStrIntCorrespondence)
    print(data['income'])

    print(data.head(10))

    plt.xlabel("Income Level")
    plt.ylabel("Number of People")
    plt.title("Population vs Income above and below 50K")
    plt.hist(data['income'], bins=2, rwidth=1, color='b')
    plt.show()


    print("1st 10 Rows of Data: \n" + str(data.head(10)) + "\n")  #Prints top 5 rows of the table

    dataX = data.drop('income', axis=1)
    dataY = data['income']
    xTrain, xTest, yTrain, yTest = train_test_split(dataX, dataY, test_size = 0.25, shuffle = True)

    global sc
    # sc = StandardScaler()     #Linear Scale
    # sc = RobustScaler()       #Moves Outliers Closer
    # sc = MaxAbsScaler()       #Scaling Sparse Data
    # sc = QuantileTransformer()  #Non-linear Scale to uniform distribution between 0 and 1
    sc = PowerTransformer()     #Non-linear Scale to Gaussian Distribution
    xTrain = sc.fit_transform(xTrain)  #Find the parameters to scale the training data around 0
    xTest = sc.transform(xTest)     #Apply the same scaling using the same parameters to testing data

    # print(xTrain[:10])

    classifier = LogisticRegression(max_iter = 12000)
    # classifier = RandomForestClassifier(n_estimators = 200)
    classifier.fit(xTrain, yTrain)
    preds = classifier.predict(xTest)

    print(classification_report(yTest, preds))

    plot_confusion_matrix(classifier, xTest, yTest)
    plt.xlabel("True  <=50K              True  >50K")
    plt.ylabel("Pred  >50K              Pred  <=50K")
    plt.title("Confusion Matrix")
    plt.show()



    global mlpc
    mlpc = MLPClassifier(hidden_layer_sizes=(16, 8, 4), max_iter=10000)
    mlpc.fit(xTrain, yTrain)
    predMlpc = mlpc.predict(xTest)

    print(classification_report(yTest, predMlpc))

    plot_confusion_matrix(mlpc, xTest, yTest)
    plt.xlabel("True  <=50K              True  >50K")
    plt.ylabel("Pred  >50K              Pred  <=50K")
    plt.title("Confusion Matrix")
    plt.show()

    cm = accuracy_score(yTest, predMlpc)
    print("Accuracy Score: " + str(cm))



    predictNew(18, "Private", 168288, "HS-grad", 9, "Never-married", "Handlers-cleaners", "Own-child", "White", "Male", 0, 0, 40, "United-States")
    predictNew(42, "Self-emp-not-inc", 174216, "Prof-school", 15, "Married-civ-spouse", "Prof-specialty", "Husband", "White", "Male", 0, 0, 38, "United-States")

def convertStrInt(column, possibleValues):
    newColumn = []
    for feature in column:
        if feature not in possibleValues:
            possibleValues.append(feature)
            newColumn.append(len(possibleValues) - 1)
        else:
            newColumn.append(possibleValues.index(feature))
    # print("len(column): " + str(len(column)))
    return newColumn

def convertNewStrInt(str, list):
    try:
        str = list.index(str)
    except:
        str = len(list)
    return str

def predictNew(age, workclass, fnlwgt, education, educationalNum, maritalStatus, occupation, relationship, race, gender, captialGain, capitalLoss, hoursPerWeek, nativeCountry):
    data = [age, workclass, fnlwgt, education, educationalNum, maritalStatus, occupation, relationship, race, gender, captialGain, capitalLoss, hoursPerWeek, nativeCountry]
    data[1] = convertNewStrInt(workclass, workclassStrIntCorrespondence)
    data[3] = convertNewStrInt(workclass, educationStrIntCorrespondence)
    data[5] = convertNewStrInt(workclass, maritalStatusStrIntCorrespondence)
    data[6] = convertNewStrInt(workclass, occupationStrIntCorrespondence)
    data[7] = convertNewStrInt(workclass, relationshipStrIntCorrespondence)
    data[8] = convertNewStrInt(workclass, raceStrIntCorrespondence)
    data[9] = convertNewStrInt(workclass, genderStrIntCorrespondence)
    data[13] = convertNewStrInt(workclass, nativeCountryStrIntCorrespondence)
    # print(data)
    data = [data]

    data = sc.fit_transform(data)
    prediction = mlpc.predict(data)

    try:
        print("\n" + str(incomeStrIntCorrespondence[int(prediction[0])]))
    except:
        print("Prediction failed")


if __name__ == "__main__":  #Run the main function
    main()
