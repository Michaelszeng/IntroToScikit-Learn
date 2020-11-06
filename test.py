# python 3.7
# Scikit-learn ver. 0.23.2
from sklearn.linear_model import LogisticRegression
from scipy.special import expit
# matplotlib 3.3.1
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report, plot_confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder, scale
from sklearn.model_selection import train_test_split

def main():
    data = pd.read_csv('adult.csv', sep=',')

    #Preprocessing
    data = data.head(400)   #Only consider the first 400 entries (so my computer doesn't just die)
    data = data[data['workclass'] != "?"]
    data = data[data['occupation'] != "?"]
    data = data[data['native-country'] != "?"]
    data['workclass'] = data['workclass'].astype('str')

    # data['fnlwgt']

    numericIncome = []
    for i in data['income']:
        if "<=50K" in i:
            numericIncome.append(0)
        elif ">50K" in i:
            numericIncome.append(1)
    data['income'] = numericIncome

    # print("1st 5 Rows of Data: \n" + str(data.head()) + "\n")  #Prints top 5 rows of the table
    print("Feature Datatype Info: \n" + str(data.info()) + "\n")  #Prints info about each column's datatype, etc
    # print("Number of Null Values Per Feature: \n" + str(data.isnull().sum()) + "\n")  #Prints how many null values there are for each column
    print(str(data['income'].value_counts()) + "\n")    #Prints a table with just the data number and the corresponding income value

    #Converting String Values to Integer
    data['workclass'], workclassStrIntCorrespondence = convertStrInt(data['workclass'])
    data['education'], educationStrIntCorrespondence = convertStrInt(data['education'])
    data['marital-status'], maritalStatusStrIntCorrespondence = convertStrInt(data['marital-status'])
    data['occupation'], occupationStrIntCorrespondence = convertStrInt(data['occupation'])
    data['relationship'], relationshipStrIntCorrespondence = convertStrInt(data['relationship'])
    data['race'], raceStrIntCorrespondence = convertStrInt(data['race'])
    data['gender'], genderStrIntCorrespondence = convertStrInt(data['gender'])
    data['native-country'], nativeCountryStrIntCorrespondence = convertStrInt(data['native-country'])

    print("workclassStrIntCorrespondence: " + str(workclassStrIntCorrespondence))
    print(data.head(10))

    plt.xlabel("Income Level")
    plt.ylabel("Number of People")
    plt.title("Income level above 50K and below")
    plt.hist(data['income'], bins=2, rwidth=1, color='b')
    plt.show()


    print("1st 10 Rows of Data: \n" + str(data.head(10)) + "\n")  #Prints top 5 rows of the table

    dataX = data.drop('income', axis=1)
    dataY = data['income']
    xTrain, xTest, yTrain, yTest = train_test_split(dataX, dataY, test_size = 0.25, shuffle = True)

    sc = StandardScaler()
    xTrain = sc.fit_transform(xTrain)  #Find the parameters to scale the training data around 0
    xTest = sc.transform(xTest)     #Apply the same scaling using the same parameters to testing data

    print(xTrain[:10])

    # classifier = LogisticRegression(max_iter = 10000)
    classifier = RandomForestClassifier(n_estimators = 200)
    classifier.fit(xTrain, yTrain)
    preds = classifier.predict(xTest)

    correct = 0
    incorrect = 0
    #Loop through the predictions and the answers
    for pred, gt in zip(preds, yTest):
        if pred == gt:
            correct += 1
        else:
            incorrect += 1
    print(f"Correct: {correct}, Incorrect: {incorrect}, % Correct: {correct/(correct + incorrect): 5.2}")

    plot_confusion_matrix(classifier, xTest, yTest)
    plt.show()

def convertStrInt(column):
    newColumn = []
    featureValues = []
    for feature in column:
        if feature not in featureValues:
            featureValues.append(feature)
            newColumn.append(len(featureValues) - 1)
        else:
            newColumn.append(featureValues.index(feature))
    # print("len(column): " + str(len(column)))
    return newColumn, featureValues

if __name__ == "__main__":  #Run the main function
    main()
