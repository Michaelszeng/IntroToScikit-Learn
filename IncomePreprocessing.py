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

def main():
    data = pd.read_csv('adult.csv', sep=',')

    workclassStrIntCorrespondence = findAllValues(data['workclass'])
    educationStrIntCorrespondence = findAllValues(data['education'])
    maritalStatusStrIntCorrespondence = findAllValues(data['marital-status'])
    occupationStrIntCorrespondence = findAllValues(data['occupation'])
    relationshipStrIntCorrespondence = findAllValues(data['relationship'])
    raceStrIntCorrespondence = findAllValues(data['race'])
    genderStrIntCorrespondence = findAllValues(data['gender'])
    nativeCountryStrIntCorrespondence = findAllValues(data['native-country'])

    print("workclassStrIntCorrespondence: " + str(workclassStrIntCorrespondence))
    print("educationStrIntCorrespondence: " + str(educationStrIntCorrespondence))
    print("maritalStatusStrIntCorrespondence: " + str(maritalStatusStrIntCorrespondence))
    print("occupationStrIntCorrespondence: " + str(occupationStrIntCorrespondence))
    print("relationshipStrIntCorrespondence: " + str(relationshipStrIntCorrespondence))
    print("raceStrIntCorrespondence: " + str(raceStrIntCorrespondence))
    print("genderStrIntCorrespondence: " + str(genderStrIntCorrespondence))
    print("nativeCountryStrIntCorrespondence: " + str(nativeCountryStrIntCorrespondence))

def findAllValues(column):
    newColumn = []
    featureValues = []
    for feature in column:
        if feature not in featureValues:
            featureValues.append(feature)
            newColumn.append(len(featureValues) - 1)
        else:
            newColumn.append(featureValues.index(feature))
    # print("len(column): " + str(len(column)))
    return featureValues

if __name__ == "__main__":  #Run the main function
    main()
