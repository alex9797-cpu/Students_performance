import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



students_data=pd.read_csv(filepath_or_buffer='student-mat.csv',sep=';')


print(f' The dataframe contains {students_data.shape[0]} observations and {students_data.shape[1]} variables')
print('\n')
colnames=students_data.columns

print('The names of the varaibales are :')
for col in colnames:
    print(col)


print('\n')
print('Datatypes of the Variables')
for col in colnames:
    print(f'Variable:  {col} has Datatype :{type(students_data[col][0]) } ')

print('\n')


# Drop Grade1 and Grade 2
students_data=students_data.drop(columns=['G1','G2'])


numeric_features=['age','Medu','Fedu','traveltime','studytime','failures','famrel','freetime','goout','Dalc','Walc','health','absences','G3']

df_numeric=students_data[numeric_features]
print(f' Dataset contains {df_numeric.shape[1]} numeric features')

# Perform descriptive analysis for the numeric variables.

for variable in df_numeric:
    print(f' Variable {variable}')
    print(f' Mean { np.mean(df_numeric[variable])} ')
    print(f' Median {np.median(df_numeric[variable])} ')
    print(f' Maximum {np.max(df_numeric[variable])}')
    print(f' Minimum {np.min(df_numeric[variable])}')
    print(f' Variance {np.std(df_numeric[variable])**2} ')
    print(f' Std {np.std(df_numeric[variable])} ')
    print('\n')


print(df_numeric.describe())


# Compute correlation matrix
corr_mat=df_numeric.corr()
# Print the correlation matrix:
print(corr_mat)

print(np.corrcoef(df_numeric['G3'],df_numeric['age'])[0,1])

# Compute correlations with the outcome variable:
abs_corr=[ np.abs(np.corrcoef(df_numeric['G3'],df_numeric[feature])[0,1])  for feature in numeric_features]
# Sort correlations:#
print(f' Absolute Value of Correlation with the Outcome Variable:')
for i in range(len(numeric_features)):
    print(numeric_features[i])
    print(abs_corr[i])
    print('\n')


# Failures and eductaion of parents have heighest correlation with the response
# varibale but also not very high


# Visualize numeric features with histogramns:

for variable in numeric_features:
    plt.hist(df_numeric[variable])
    plt.title(f'Histogram of {variable}')
    plt.xlabel(variable)
    plt.ylabel('Frequency')
    plt.show()

#  Create boxplots for numeric features:

for variable in numeric_features:
    plt.boxplot(df_numeric[variable])
    plt.title(f'Boxplot of {variable}')
    plt.show()



cat_features=['school','sex','address','famsize','Pstatus','Mjob', 'Fjob', 'reason', 'guardian','schoolsup', 'famsup', 'paid', 'activities', 'nursery','higher', 'internet']

df_cat=students_data[cat_features]

print(students_data['school'].value_counts()/len(students_data['school']))

# Visualize Relative Frequencies of Variables

for col in df_cat.columns:
    plt.bar(df_cat[col].value_counts()/len(df_cat[col]))
    plt.show()




































