# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import Necessary Libraries and Load Data

2.Split Dataset into Training and Testing Sets

3.Train the Model Using Stochastic Gradient Descent (SGD)

4.Make Predictions and Evaluate Accuracy

5.Generate Confusion Matrix

## Program:
```PYTHON
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: VARUN J C
RegisterNumber:  212224240179
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 

#load the dataset
iris = load_iris()

#create pandas dataframe
df=pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target']=iris.target

print(df.head())

#split the data into features x and target y
x=df.drop('target',axis=1)
y= df['target']

#split the data into training and testing set
x_train, x_test,y_train,y_test = train_test_split(x,y, test_size = 0.2, random_state = 42)

sgd_clf = SGDClassifier(max_iter = 1000, tol= 1e-3)#create sgd classifier with default parameter
sgd_clf.fit(x_train, y_train)#train the classifier on the training data

y_pred = sgd_clf.predict(x_test)#make predictions on the testing data

accuracy = accuracy_score(y_test, y_pred)# evaluate the classifier's accuracy
print(f"Accuracy:{accuracy:.3f}")

#calculate the confusion matrix
cf=confusion_matrix(y_test, y_pred)
print("Confusion Matrix")
print(cf)

*/
```

## Output:
![image](https://github.com/user-attachments/assets/7ccbe75b-b146-4557-8a84-9e193711833d)



## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
