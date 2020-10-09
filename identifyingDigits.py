# python 3.7
# Scikit-learn ver. 0.23.2
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
from scipy.special import expit
# matplotlib 3.3.1
from matplotlib import pyplot

digits = load_digits()  #Load Data Library
#digitsX = all the images
digitsX = digits.images.reshape(len(digits.images), 64)
#digitsY = all the answers
digitsY = digits.target

#Split data, some for training, some for testing
#30% of the total data will be for testing
trainX, testX, trainY, testY = train_test_split(digitsX, digitsY, test_size = 0.3, shuffle = True)

classifier = LogisticRegression(max_iter = 10000)
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


#Attemping to Plot
pyplot.figure(1, figsize=(4, 3))
pyplot.clf()
pyplot.scatter(X.ravel(), y, color='black', zorder=20)
X_test = np.linspace(-5, 10, 300)

loss = expit(X_test * clf.coef_ + clf.intercept_).ravel()
pyplot.plot(X_test, loss, color='red', linewidth=3)

ols = linear_model.LinearRegression()
ols.fit(X, y)
pyplot.plot(X_test, ols.coef_ * X_test + ols.intercept_, linewidth=1)
pyplot.axhline(.5, color='.5')

pyplot.ylabel('y')
pyplot.xlabel('X')
pyplot.xticks(range(-5, 10))
pyplot.yticks([0, 0.5, 1])
pyplot.ylim(-.25, 1.25)
pyplot.xlim(-4, 10)
pyplot.legend(('Logistic Regression Model', 'Linear Regression Model'),
           loc="lower right", fontsize='small')
pyplot.tight_layout()
pyplot.show()
