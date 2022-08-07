
# Perform Regression on the ourcome Grade 3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import concat_dummy , metrics_table
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error , mean_absolute_error, max_error

# Import Data
students_data=pd.read_csv(filepath_or_buffer='student-mat.csv',sep=';')

print(f' Dimension before dropping NA values: {students_data.shape}')
print('\n')
students_data.dropna(inplace=True)

print(f' Dimension after dropping NA values: {students_data.shape}')

# Drop Grades which we do not need:

students_data.drop(columns=['G1','G2'], inplace=True)

# Change all categrocial variables to dummy variables:

cat_features=['school','sex','address','famsize','Pstatus','Mjob', 'Fjob', 'reason', 'guardian','schoolsup', 'famsup', 'paid', 'activities', 'nursery','romantic','higher', 'internet']


for variable in cat_features:
    students_data=concat_dummy(students_data,colname=variable)


# Perform train test split with putting 20% into the test dataset

print(' Perform Train Test Split:')
X=students_data.drop(columns='G3')
y=students_data['G3']

X_train,X_test, y_train,y_test= train_test_split(X,y,test_size=0.25,random_state=123)

print(f' Number of training  instances {X_train.shape[0]}')
print(f' Number of test  instances {X_test.shape[0]}')

# Perform feature scaling on train and test set:

numeric_features=['age','Medu','Fedu','traveltime','studytime','failures','famrel','freetime','goout','Dalc','Walc','health','absences']

scaler=StandardScaler()

X_train[numeric_features]=scaler.fit(X_train[numeric_features]).transform(X_train[numeric_features])
X_test[numeric_features]=scaler.fit(X_test[numeric_features]).transform(X_test[numeric_features])

# Fit a Linear Regression model:
linear_model=LinearRegression()
# Perform 10 fold cross validation

scores=cross_val_score(linear_model, X_train, y_train, cv=5,scoring='neg_mean_squared_error')
print(f' Cross Validation results Linear Model{scores}')
print('\n')

# Fit A Regression Tree_:

tree_model=DecisionTreeRegressor(random_state=0)

scores=cross_val_score(tree_model, X_train, y_train, cv=5,scoring='neg_mean_squared_error')
print(f' Cross Validation results Regression Tree {scores}')
print('\n')



# Check Performance on Testset:
linear_model.fit(X_train,y_train)
y_pred1=linear_model.predict(X_test)


lm_results=  metrics_table(y_test,y_pred1)

print(lm_results)




























































