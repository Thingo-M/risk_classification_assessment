import kagglehub
import os
import pandas as pd
import numpy as np
import regex as re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import metrics

# Download latest version
#path = kagglehub.dataset_download("wordsforthewise/lending-club")

#print("Path to dataset files:", path)

os.chdir(r"C:\Users\mafaesami\OneDrive - University of the Free State\Github\Projects\dataset\accepted_2007_to_2018q4")
df_accept = pd.read_csv('accepted_2007_to_2018Q4.csv')

#os.chdir(r"C:\Users\mafaesami\.cache\kagglehub\datasets\wordsforthewise\lending-club\versions\3\rejected_2007_to_2018q4.csv")
#df_reject = pd.read_csv('rejected_2007_to_2018Q4.csv')

#os.chdir(r"C:\Users\mafaesami\.cache\kagglehub\datasets\wordsforthewise\lending-club\versions\3")
#diction = pd.read_excel('LCDataDictionary.xlsx')
#col = pd.read_excel('Accept Columns - Copy.xlsx')

#col2 = pd.merge(col,diction,on='LoanStatNew', how='left')
#col2.to_excel('Column Names.xlsx')
#accept_col = pd.Series(list(df_accept.columns))
#accept_col.to_excel('Accept Columns.xlsx')

#reject_col = pd.Series(list(df_reject.columns))
#reject_col.to_excel('Reject Columns.xlsx')


def clean_id(value):
    if isinstance(value, int):
        return value
    elif isinstance(value, str) and value.isdigit():
        return int(value)
    return None

df_accept['id'] = df_accept['id'].apply(clean_id)
df_accept = df_accept[pd.notna(df_accept['id'])] 

##Credit Score
a = df_accept['fico_range_low'] - df_accept['last_fico_range_low']
min(a)
max(a)

#Debt-to-income ratio
df_accept['dti']

#Revolving Line Credit Utilisation
df_accept['revol_util']
#df_accept['tot_hi_cred_lim']
#df_accept['total_bc_limit']
#df_accept['total_il_high_credit_limit']

df_accept['loan_status'].unique()

df_accept['default'] = np.nan

df_accept['default'][df_accept['loan_status'] == 'Fully Paid'] = 0
df_accept['default'][df_accept['loan_status'] == 'Current'] = 0
df_accept['default'][df_accept['loan_status'] == 'Late (31-120 days)'] = 0
df_accept['default'][df_accept['loan_status'] == 'Late (16-30 days)'] = 0
df_accept['default'][df_accept['loan_status'] == 'Does not meet the credit policy. Status:Fully Paid'] = 0

df_accept['default'][df_accept['loan_status'] == 'Charged Off'] = 1
df_accept['default'][df_accept['loan_status'] == 'In Grace Period'] = 1
df_accept['default'][df_accept['loan_status'] == 'Default'] = 1
df_accept['default'][df_accept['loan_status'] == 'Does not meet the credit policy. Status:Charged Off'] = 1


df_accept = df_accept[pd.notna(df_accept['dti'])]
df_accept = df_accept[pd.notna(df_accept['revol_util'])]

df_accept['last_fico_range_low']

df_accept[pd.isna(df_accept['default'])]



X = df_accept[['last_fico_range_low','dti','revol_util']]
y = df_accept['default']


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict probabilities of default
probabilities = model.predict_proba(X_test)[:, 1]  # Probabilities of default (class 1)
predictions = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy * 100:.2f}%")


from sklearn.metrics import f1_score
f1_score(y_test, predictions, average='micro')

# Display the probabilities
#print("\nPredicted Probabilities of Default:")
#for i, prob in enumerate(probabilities):
#    print(f"Instance {i + 1}: {prob * 100:.2f}%")
