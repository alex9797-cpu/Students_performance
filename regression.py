
# Perform Regression on the ourcome Grade 3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import concat_dummy
from sklearn.model_selection import train_test_split

# Import Data
students_data=pd.read_csv(filepath_or_buffer='student-mat.csv',sep=';')

print(f' Dimension before dropping NA values: {students_data.shape}')
print('\n')
students_data.dropna(inplace=True)

print(f' Dimension after dropping NA values: {students_data.shape}')

# Drop Grades which we do not need:

students_data.drop(columns=['G1','G2'], inplace=True)

# Change all categrocial variables to dummy variables:

cat_features=['school','sex','address','famsize','Pstatus','Mjob', 'Fjob', 'reason', 'guardian','schoolsup', 'famsup', 'paid', 'activities', 'nursery','higher', 'internet']


for variable in cat_features:
    students_data=concat_dummy(students_data,colname=variable)


# Perform train test split with putting 20% into the test dataset

print(' Perform Train Test Split:')
X=students_data.drop(columns='G3')
y=students_data['G3']

X_train,X_test, y_train,y_test= train_test_split(X,y,test_size=0.25,random_state=123)

print(f' Number of training  instances {X_train.shape[0]}')
print(f' Number of test  instances {X_test.shape[0]}')
























