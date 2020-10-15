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
print("1st 5 Rows of Data: \n" + str(wine.head()) + "\n")  #Prints top 5 rows of the table
print("Feature Datatype Info: \n" + str(wine.info()) + "\n")  #Prints info about each column's datatype, etc
print("Number of Null Values Per Feature: \n" + str(wine.isnull().sum()) + "\n")  #Prints how many null values there are for each column
