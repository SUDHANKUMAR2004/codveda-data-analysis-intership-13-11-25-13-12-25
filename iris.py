#task1 basic
import pandas as pd
iris_data=pd.read_csv("E:\\Internship codeveda\\dataset\\iris.csv")
iris_data.dropna(inplace=True)  #remove missing values
iris_data=iris_data.drop_duplicates() #delete duplicates
#print(iris_data)

#task2
summary=iris_data.describe()
#print(summary)
#print(iris_data.nunique()) #to find unique values
#for data visualization we can use seaborn and matplotlib
import seaborn as sbn
import matplotlib.pyplot as plt
num_data=iris_data.select_dtypes(include=['number'])
sbn.heatmap(num_data.corr(), annot=True)
#plt.show()

#histogram
sbn.histplot(iris_data['petal_length'],kde=True)
plt.title("Histogram of petal length")
plt.xlabel("petal length")
plt.ylabel("frequency")
#plt.show()

#boxplot
sbn.boxplot(data=iris_data[['sepal_length','sepal_width','petal_length','petal_width']])
plt.title("box plot of iris")
plt.xlabel("features")
plt.ylabel("measurement in cm")
#plt.show()

#scatterplot
sbn.scatterplot(x='sepal_length',y='petal_length',hue='species',data=iris_data)
plt.title("scatter plot  sepal length vs petal length")
plt.xlabel("sepal length")
plt.ylabel('petal length')
#plt.show()

#task 3
#bar plot
species_count=iris_data['species'].value_counts()

plt.bar(species_count.index,species_count.values)
plt.xlabel("spices")
plt.ylabel('count')
plt.title('count of each iris species')
#plt.show()

#line chart
plt.plot(iris_data['sepal_length'])
plt.title("trend of sepal length")
plt.xlabel("index")
plt.ylabel("sepal length in cm")
#plt.savefig("line_chart.png")
#plt.show()

#intermediate task 1-regression analysis
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error

#we use x and y variables for analysis
x=iris_data[['sepal_length']]
y=iris_data[['petal_length']]

#train test split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

#create and fit linear regression model
model=LinearRegression()
model.fit(x_train,y_train)

#make prediction
y_pred=model.predict(x_test)

#evaluate the model
r2=r2_score(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)
#print("r-squared: ",r2)
#print("mean squared error: ",mse)

#interpret the coefficients
#print('slope (coefficient): ',model.coef_[0])
#print("intercept: ",model.intercept_)

#task2 time series analysis, for this i taken stock dataset.
#because the time series analysis needs time column
from statsmodels.tsa.seasonal import seasonal_decompose

stock=pd.read_csv("E:\\Internship codeveda\\dataset\\StockPricesDataSet.csv")
stock.dropna(inplace=True)  #remove missing values
stock=stock.drop_duplicates() #delete duplicates

#convert date column to datatime

stock['date']=pd.to_datetime(stock['date'])


#set date as index
stock.set_index('date',inplace=True)

#plot the original time series using matplotlib
plt.figure(figsize=(10,5))
plt.plot(stock['high'])
plt.title('original time series')
plt.xlabel('date')
plt.ylabel('value')
#plt.show()

#decompose the time series-trend,seasonality,residuals
decomp=seasonal_decompose(stock['high'],model='additive',period=365)


plt.figure(figsize=(10,8))
decomp.plot()
#plt.show()

#moving average smoothing
stock['moving_avg_7']=stock['high'].rolling(window=7).mean() #weekly smoothing
stock['moving_avg_30']=stock['high'].rolling(window=30).mean() #monthly smoothing

#plot the smoothed curves
plt.figure(figsize=(10,5))
plt.plot(stock['high'],label='original')
plt.plot(stock['moving_avg_7'],label='weekly moving average')
plt.plot(stock['moving_avg_30'],label='monthly moving average')
plt.title('moving average smoothing')
plt.xlabel('date')
plt.ylabel('high')
#plt.show()


#task 3-clustering analysis- k-means algorithm
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

#select numeric column for clustering
numdata=iris_data.select_dtypes(include=['number'])

#standardize the data
scalar=StandardScaler()
scaled_data=scalar.fit_transform(numdata)

#determine the optimal number of cluster using elbow method
inertia_values=[]

for k in range(1,11):
    kmeans=KMeans(n_clusters=k,random_state=42)
    kmeans.fit(scaled_data)
    inertia_values.append(kmeans.inertia_)

#plot the elbow curve
plt.figure(figsize=(8,5))
plt.plot(range(1,11),inertia_values,marker=0)
plt.title('elbow method to find optimal k')
plt.xlabel('number of clusters - k')
plt.ylabel('inertia')
#plt.show()

#fit the k means with k
kmeans=KMeans(n_clusters=3,random_state=42)
clusters=kmeans.fit_predict(scaled_data)

#add cluster label to dataset
iris_data['Cluster']=clusters

#visualize the cluster
plt.figure(figsize=(8,5))
sbn.scatterplot(x=scaled_data[:,0],y=scaled_data[:,1],hue=clusters,palette='Set2')
plt.title('k means clustering visualization')
plt.xlabel('feature 1 scaled')
plt.ylabel('feature 2 scaled')
#plt.show()

#Advanced task 1 predictive modelling
import numpy as np

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

#print("Best Parameters:", grid.best_params_)
best_model = grid.best_estimator_

#evalute the tuned model
best_pred = best_model.predict(x_test1)
evaluate_model(y_test1, best_pred)
