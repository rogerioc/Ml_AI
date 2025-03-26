import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB 
from sklearn.ensemble import RandomForestClassifier


def printAccuracy(name,y_test, y_pred):
    print(f"{name} -----------------")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print(accuracy_score(y_test, y_pred))

#Preprocessing
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Classifier Decision Tree
classifierDecisionTree = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifierDecisionTree.fit(X_train, y_train)
y_pred = classifierDecisionTree.predict(X_test)
printAccuracy("Decision Tree",y_test, y_pred)
 
# Classifier Logistic Regression
classifierLogistic = LogisticRegression(random_state = 0)
classifierLogistic.fit(X_train, y_train)
y_pred = classifierLogistic.predict(X_test)
printAccuracy("Logistic Regression",y_test, y_pred)

# Classifier K-N
classifierKNN = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifierKNN.fit(X_train, y_train)
y_pred = classifierKNN.predict(X_test)
printAccuracy("KNN",y_test, y_pred)

# Classifier SVM
classifierSVM = SVC(kernel = 'linear', random_state = 0)
classifierSVM.fit(X_train, y_train)
y_pred = classifierSVM.predict(X_test)
printAccuracy("SVM",y_test, y_pred)

# Classifier Kernel SVM
classifierKernelSVM = SVC(kernel = 'rbf', random_state = 0) 
classifierKernelSVM.fit(X_train, y_train)
y_pred = classifierKernelSVM.predict(X_test)
printAccuracy("Kernel SVM",y_test, y_pred)

# Classifier Naive Bayes
classifierNaiveBayes = GaussianNB()
classifierNaiveBayes.fit(X_train, y_train)
y_pred = classifierNaiveBayes.predict(X_test)
printAccuracy("Naive Bayes",y_test, y_pred)

# Classifier Random Forest
classifierRandomForest = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifierRandomForest.fit(X_train, y_train)
y_pred = classifierRandomForest.predict(X_test)
printAccuracy("Random Forest",y_test, y_pred)
