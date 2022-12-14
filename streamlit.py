import pandas as pd
import numpy as np
import os 
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
os.path.relpath(r'C:\Users\Kim\Desktop\Georgetown\Programming II\Final Project')
# os.chdir(r'C:\Users\Kim\Desktop\Georgetown\Programming II\Final Project')

s = pd.read_csv('social_media_usage.csv')
print(s)

def clean_sm(dataframe):
    dataframe = np.where(dataframe== 1,
                         1,
                         0)
    return dataframe

ss = s
ss['sm_li'] = clean_sm(ss['web1h'])
ss = ss[['sm_li', 'income', 'educ2','par', 'marital', 'gender', 'age']]
ss = ss[ss.income <= 9]
ss = ss[ss.educ2 <= 8]
ss['par'] = clean_sm(ss['par'])
ss['marital'] = clean_sm(ss['marital'])

def clean_g(dataframe):
    dataframe = np.where(dataframe== 2,
                         1,
                         0)
    return dataframe

ss['gender'] = clean_g(ss['gender']) # 1 will be female
ss = ss[ss.age <= 98]

########## Model building
feature = ss.drop("sm_li", axis=1)
vector = ss.sm_li

X_train, X_test, y_train, y_test = train_test_split(feature, vector, test_size=0.2, random_state=100)

logreg = LogisticRegression(random_state=100, class_weight = 'balanced')
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)







age_user=st.slider("What is your age", min_value=1, max_value=98,value= 49)

gender_user = st.selectbox("What's your gender?", 
              options = ["male",
                         "female",
                         "other",
                         "Don\'t know",
                         'Refuse to answer'])
if gender_user == "female":
    gender_user = 1
else :
    gender_user = 0

marital_user = st.selectbox("What is your current marital status?", 
              options = ["Married",
                         "Living with a partner",
                         "Divorced",
                         "Separated",
                         'Widowed',
                         'Never been married',
                         'Don\'t know',
                         'Refuse to answer'])
if marital_user == "Married":
    marital_user = 1
else :
    marital_user = 0

par_user = st.selectbox("Are you a parent of a child under 18 living in your home?", 
              options = ["Yes",
                         "No",
                         "Don\'t know",
                         'Refuse to answer'])
if par_user == "Yes":
    par_user = 1
else :
    par_user = 0

educ2_user = st.selectbox("What is your highest level of school/degree completed?", 
              options = ["Less than high school (Grades 1-8 or no formal schooling)",
                         "High school incomplete (Grades 9-11 or Grade 12 with NO diploma)",
                         "High school graduate (Grade 12 with diploma or GED certificate)",
                         'Some college, no degree (includes some community college)',
                         'Two-year associate degree from a college or university',
                         'Four-year college or university degree/Bachelor\’s degree (e.g., BS, BA, AB)',
                         'Some postgraduate or professional schooling, no postgraduate degree (e.g. some graduate school)',
                         'Postgraduate or professional degree, including master\’s, doctorate, medical or law degree (e.g., MA, MS, PhD, MD, JD)',
                         ])
if educ2_user == "Less than high school (Grades 1-8 or no formal schooling)":
    educ2_user = 1
elif educ2_user == "High school incomplete (Grades 9-11 or Grade 12 with NO diploma)" :
    educ2_user = 2
elif educ2_user == "High school graduate (Grade 12 with diploma or GED certificate)" :
    educ2_user = 3
elif educ2_user == 'Some college, no degree (includes some community college)' :
    educ2_user = 4
elif educ2_user == "Two-year associate degree from a college or university" :
    educ2_user = 5
elif educ2_user == "Four-year college or university degree/Bachelor\’s degree (e.g., BS, BA, AB)" :
    educ2_user = 6
elif educ2_user == "Some postgraduate or professional schooling, no postgraduate degree (e.g. some graduate school)" :
    educ2_user = 7
elif educ2_user == "Postgraduate or professional degree, including master\’s, doctorate, medical or law degree (e.g., MA, MS, PhD, MD, JD)" :
    educ2_user = 8

income_user = st.selectbox("What is your household income?", 
              options = ["Less than $10,000",
                         "10 to under $20,000",
                         "20 to under $30,000",
                         '30 to under $40,000',
                         '40 to under $50,000',
                         '50 to under $75,000',
                         '75 to under $100,000',
                         '100 to under $150,000',
                         '$150,000 or more'
                         ])
if income_user == "Less than $10,000":
    income_user = 1
elif income_user == "10 to under $20,000" :
    income_user = 2
elif income_user == "20 to under $30,000" :
    income_user = 3
elif income_user == '30 to under $40,000' :
    income_user = 4
elif income_user == "40 to under $50,000" :
    income_user = 5
elif income_user == "50 to under $75,000" :
    income_user = 6
elif income_user == "75 to under $100,000" :
    income_user = 7
elif income_user == "100 to under $150,000" :
    income_user = 8
elif income_user == "$150,000 or more" :
    income_user = 9





example = [[income_user, educ2_user, par_user, marital_user, gender_user, age_user]]
example_test = pd.DataFrame(example, columns =['income', 'educ2','par', 'marital', 'gender', 'age'])
prediction = logreg.predict(example_test)
probability = logreg.predict_proba(example_test)[0][1]
example_test['prediction'] = prediction
example_test['prediction probability'] = probability

result =[]

st.write('Is this person a Linkedin user?')
if prediction == 1:
    result = 'Yes'
else :
    result = 'No'
    
st.write(result)
    
st.write('What is the probaility that this person is a Linkedin user?')
st.write(example_test['prediction probability'])

