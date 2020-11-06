#https://www.youtube.com/watch?v=0Lt9w-BxKFQ

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

wine = pd.read_csv('winequality-red.csv', sep=';')
print(str(wine) + "\n")
print("1st 5 Rows of Data: \n" + str(wine.head()) + "\n")  #Prints top 5 rows of the table
print("Feature Datatype Info: \n" + str(wine.info()) + "\n")  #Prints info about each column's datatype, etc
print("Number of Null Values Per Feature: \n" + str(wine.isnull().sum()) + "\n")  #Prints how many null values there are for each column
# print(wine['quality'])    #Prints a table with just the data number and the corresponding quality number

bins = (2, 6.5, 8)
group_names = ['bad', 'good']
wine['quality'] = pd.cut(wine['quality'], bins = bins, labels = group_names)

label_quality = LabelEncoder()
wine['quality'] = label_quality.fit_transform(wine['quality'])
print("1st 5 Rows of Data: \n" + str(wine.head(15)) + "\n")  #Prints top 15 rows of the table

print("Good Vs Bad Wine Count: " + str(wine['quality'].value_counts) + "\n")


wineX =
wineY = label_quality.fit_transform(wine['quality'])    #gotta test this line
trainX, testX, trainY, testY = train_test_split(wineX, wineY, test_size = 0.3, shuffle = True)
