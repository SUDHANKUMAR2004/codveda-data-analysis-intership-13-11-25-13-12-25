import pandas as pd
import numpy as np
import seaborn as sbn
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

#load dataset
churn=pd.read_csv("E:\\Internship codeveda\\dataset\\churn-bigml-80.csv")

#cleaning
churn=churn.dropna()
#print(churn)

#identify the cateforical columns
cate_cols=churn.select_dtypes(include=['object']).columns

#encode using labelencoder
le=LabelEncoder()
for col in cate_cols:
    churn[col]=le.fit_transform(churn[col])

#split the features and target

x=churn.drop('Churn',axis=1)
y=churn['Churn']

#train test split
x_train1,x_test1,y_train1,y_test1=train_test_split(x,y,test_size=0.2,random_state=42)

#feature scaling
scalar1=StandardScaler()
x_train1=scalar1.fit_transform(x_train1)
x_test1=scalar1.transform(x_test1)

#model training

#logistic regression
log_model=LogisticRegression()
log_model.fit(x_train1,y_train1)
log_pred=log_model.predict(x_test1)

#decision tree
tree_model=DecisionTreeClassifier()
tree_model.fit(x_train1,y_train1)
tree_pred=tree_model.predict(x_test1)

#random forest
rf_model=RandomForestClassifier()
rf_model.fit(x_train1,y_train1)
rf_pred=rf_model.predict(x_test1)

#model evaluation function
def evaluate_model(y_test,y_pred):
    print("accuracy: ",accuracy_score(y_test,y_pred))
    print("precision: ",precision_score(y_test,y_pred))
    print("recall: ",recall_score(y_test,y_pred))
    print('f1 score: ',f1_score(y_test,y_pred))
    print('\nClassification report: \n',classification_report(y_test,y_pred))


#evaluate each model
print('logistic regression: ')
evaluate_model(y_test1,log_pred)

print('decision tree: ')
evaluate_model(y_test1,tree_pred)

print('random forest: ')
evaluate_model(y_test1,rf_pred)

#confusion matrix
cm = confusion_matrix(y_test1, rf_pred)
sbn.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("confusion matrix-random forest")
#plt.show()

#hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [5, 10, 15, None],
    'criterion': ['gini', 'entropy']
}

grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=3, scoring='accuracy')
grid.fit(x_train1, y_train1)

print("Best Parameters:", grid.best_params_)
best_model = grid.best_estimator_

#evalute the tuned model
best_pred = best_model.predict(x_test1)
evaluate_model(y_test1, best_pred)
