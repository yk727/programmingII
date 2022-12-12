import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

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

logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)







user_age=st.slider("What is your age", min_value=10, max_value=97,value= 18, step=1)
user_gender= st.selectbox("What is your gender? If female select 1, if male select 0",options=[1,0])
user_married= st.selectbox("Are you married? If so select 1, if not select 0",options=[1,0])
user_parent= st.selectbox("Are you a parent? If so select 1, if not select 0",options=[1,0])
st.markdown("What is your education. 1= Less than High School, 2= HS incomplete, 3=HS diploma, 4= some college, 5= two year college, 6= 4 year college, 7= post graduate schooling, 8=PHD")
user_education= st.slider("What is your highest education", min_value=1, max_value=8,value= 1, step=1)
st.markdown("What is your income 1= Less than 10,000 2= 10 to under 20,000 3= 20 to under 30,000 4= 30 to under 40,000 5= 40 to under 50,000 6=50 to under 60,000 7=70 to under 100,000 8=100 to under 150,000 9= more than 150,000")
user_income= st.slider("What is your income", min_value=1, max_value=9,value= 1, step=1)



example = [[8, 7, 0, 1, 1, 42]]
example_test = pd.DataFrame(example, columns =['income', 'educ2','par', 'marital', 'gender', 'age'])
prediction = logreg.predict(example_test)
probability = logreg.predict_proba(example_test)[0][1]
example_test['prediction'] = prediction
example_test['prediction probability'] = probability
example_test