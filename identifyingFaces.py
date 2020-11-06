# python 3.7
# Scikit-learn ver. 0.23.2
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
from scipy.special import expit
# matplotlib 3.3.1
from matplotlib import pyplot

import time
start_time = time.time()

data = fetch_olivetti_faces()  #Load Data Library
#digitsX = all the images
dataX = data.images.reshape(len(data.images), 4096)
#digitsY = all the answers
dataY = data.target

#Split data, some for training, some for testing
#30% of the total data will be for testing
trainX, testX, trainY, testY = train_test_split(dataX, dataY, test_size = 0.3, shuffle = True)

classifier = SGDClassifier(max_iter = 10000)
# classifier = RidgeClassifier(max_iter = 10000)
# classifier = LogisticRegression(max_iter = 10000)
classifier.fit(trainX, trainY)
preds = classifier.predict(testX)

correct = 0
incorrect = 0
#Loop through the predictions and the answers
for pred, gt in zip(preds, testY):
    if pred == gt:
        correct += 1
    else:
        incorrect += 1
print(f"Correct: {correct}, Incorrect: {incorrect}, % Correct: {correct/(correct + incorrect): 5.2}")

plot_confusion_matrix(classifier, testX, testY)
pyplot.show()
print("--- %s seconds ---" % (time.time() - start_time))
