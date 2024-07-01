# library yang diperlukan
from pandas import read_csv
from sklearn.datasets import make_classification
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
from sklearn import preprocessing
import pickle

# DATA PREPROCESSING
# open file
filename = "loan_data_2007_2014.csv"
dataset = pd.read_csv(filename, low_memory=False)

# check for anything wrong with the file
print(dataset.shape) # melihat dimensi file
print(dataset.head()) # cek isi filenya
print(dataset.duplicated().any())  # cek duplicated values (tidak ada)
dataset = dataset.drop(  #drop useless attributes
    ['Unnamed: 0', 'inq_last_12m', 'id', 'member_id', 'loan_amnt', 'funded_amnt_inv', 'emp_title', 'issue_d',
     'pymnt_plan', 'url', 'desc', 'title', 'zip_code',
     'delinq_2yrs', 'earliest_cr_line', 'inq_last_6mths', 'mths_since_last_delinq', 'purpose', 'mths_since_last_record',
     'total_cu_tl', 'inq_fi', 'total_rev_hi_lim', 'all_util', 'max_bal_bc', 'open_rv_24m', 'il_util', 'total_bal_il',
     'mths_since_rcnt_il',
     'open_il_24m', 'open_il_12m', 'open_il_6m', 'open_acc_6m', 'tot_cur_bal', 'verification_status_joint',
     'acc_now_delinq', 'dti_joint',
     'policy_code', 'mths_since_last_major_derog', 'collections_12_mths_ex_med', 'application_type', 'sub_grade',
     'initial_list_status',
     'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int', 'last_pymnt_d', 'last_pymnt_amnt',
     'last_credit_pull_d',
     'initial_list_status', 'out_prncp', 'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int',
     'collection_recovery_fee', 'recoveries',
     'next_pymnt_d', 'last_credit_pull_d', 'open_rv_12m'], axis=1)
print(dataset.isnull().sum())  # check if there is N/A values
print(dataset.info())  # check type of data

# EDA
print(dataset.describe())

#histograms
warnings.filterwarnings("ignore")
for data in dataset.select_dtypes(include="number").columns:
    sns.histplot(data=dataset, x=data)
    plt.show()

#boxplots for outliers
for data2 in dataset.select_dtypes(include="number").columns:
    sns.boxplot(data=dataset, x=data2)
    plt.show()

# dealing with missing values
dataset = dataset.drop(['annual_inc_joint', 'emp_length', 'tot_coll_amt'], axis=1)  #too many missing values
dataset = dataset.dropna(subset=['annual_inc', 'open_acc', 'pub_rec', 'revol_util', 'total_acc'])
print("_______________")
print(dataset.isnull().sum())

# removing outliers (only on numerical data, while leaving alone descriptive data)
# not working yet
"""
def outlierRemover(col):
    q1, q3 = np.percentile(col, [25, 75])
    iqr = q3 - q1
    lb = q1 - 1.5 * iqr
    ub = q3 + 1.5 * iqr
    return lb, ub

for i in ['funded_amnt', 'int_rate', 'installment', 
'annual_inc','dti', 'open_acc', 'pub_rec', 'revol_bal', 
'revol_util', 'total_acc', 'out_prncp_inv', 
'total_rec_late_fee']:
    lb, ub = outlierRemover(dataset[i])
    dataset[i] = np.where(dataset[i] < lb, lb, dataset[i])
    dataset[i] = np.where(dataset[i] > ub, ub, dataset[i])
"""

# separating numerical and explanatory values
numerical = dataset.select_dtypes(exclude=['object'])
categorical = dataset.select_dtypes(include=['object'])

encode = preprocessing.LabelEncoder()
label_encoded_categorical = categorical.apply(encode.fit_transform)

#putting it back together
df = pd.concat([numerical, label_encoded_categorical], axis=1)

#changing grade to 0 and 1
grade = df['grade']
def changeGrade(score):
    if score in [0, 1, 2]:
        return 1
    else:
        return 0


new_grade = grade.apply(changeGrade)
df['grade'] = new_grade

#seperating dependent and independent variable
x = df.drop(['grade'], axis=1)
grade = df.grade

logit_model = sm.Logit(grade, x)
result = logit_model.fit()

print(result.summary2())

# train the model
x_train, x_test, grade_train, grade_test = train_test_split(x, grade, test_size=0.3, random_state=100)
model = LogisticRegression(penalty='l1', C=0.01, solver='liblinear')
model.fit(x_train, grade_train)
print(f"Accuracy: {model.score(x_test, grade_test)}")
grade_prediction = model.predict(x_test)
print(classification_report(grade_test, grade_prediction))

#saving the machine learning model
with open('modelledData.pkl', 'wb') as file:
    pickle.dump(model, file)

with open('modelledData.pkl', 'rb') as file:
    clf = pickle.load(file)

finalResult = clf.score(x_test, grade_test)
print('________')
print(finalResult)
