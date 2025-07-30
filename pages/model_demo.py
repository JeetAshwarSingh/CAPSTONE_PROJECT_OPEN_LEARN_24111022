import streamlit as st
import pickle 
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,roc_auc_score

df = pd.read_csv('cleaned_survey.csv')
df=df[df['seek_help']!="Don't know"]
df['seek_help']=df['seek_help'].map({'No':0,'Yes':1})

x = df.drop(columns=['seek_help'])
y = df['seek_help']
x_train,x_test,y_train,y_test = train_test_split( x , y , train_size=0.8,random_state=42 )

with open('classification_xgb.pkl','rb') as f:
    model_xgb =pickle.load(f)

st.title('MODEL DEMO')
st.header(':blue[XGBOOST CLASSIFIER DEMO]',divider='blue')
st.write("___since XGB classifier is the best model for classification of employee's into whether they have mental health issue or not therefore demo of it is provided in this section___")
Age = st.slider('Age',18,70)
Country = st.selectbox('Country',['United States', 'Canada', 'United Kingdom', 'Bulgaria', 'France', 'Portugal', 'Netherlands', 'Switzerland', 'Poland', 'Australia', 'Russia', 'Mexico', 'Brazil', 'Slovenia', 'Costa Rica', 'Austria', 'Ireland', 'India', 'South Africa', 'Germany', 'Italy', 'Sweden', 'Colombia', 'Latvia', 'Romania', 'Belgium', 'New Zealand', 'Spain', 'Finland', 'Uruguay', 'Israel', 'Hungary', 'Singapore', 'Japan', 'Nigeria', 'Croatia', 'Norway', 'Thailand', 'Denmark', 'Greece', 'Moldova', 'Georgia', 'China', 'Czech Republic', 'Philippines'])
self_employed = st.selectbox('self_employed',['not sure', 'Yes', 'No'])
family_history = st.selectbox('family_history',['No', 'Yes'])
treatment = st.selectbox('treatment',['Yes', 'No'])
work_interfere = st.selectbox('work_interfere',['Often', 'Rarely', 'Sometimes', 'Never', 'dont know'])
no_employees = st.selectbox('no_employees',['6-25', '26-100', '1-5', '100-500', '500-1000', 'More than 1000'])
remote_work = st.selectbox('remote_work',['No', 'Yes'])
tech_company = st.selectbox('tech_company',['Yes', 'No'])
benefits = st.selectbox('benefits',['Yes', 'No', "Don't know"])
care_options = st.selectbox('care_options',['Not sure', 'No', 'Yes'])
wellness_program = st.selectbox('wellness_program',['No', 'Yes', "Don't know"])
anonymity = st.selectbox('anonymity',['Yes', "Don't know", 'No'])
leave = st.selectbox('leave',['Somewhat easy', 'Somewhat difficult', "Don't know", 'Very difficult', 'Very easy'])
mental_health_consequence = st.selectbox('mental_health_consequence',['No', 'Yes', 'Maybe'])
phys_health_consequence = st.selectbox('phys_health_consequence',['No', 'Yes', 'Maybe'])
coworkers = st.selectbox('coworkers',['Some of them', 'Yes', 'No'])
supervisor = st.selectbox('supervisor',['Yes', 'No', 'Some of them'])
mental_health_interview = st.selectbox('mental_health_interview',['No', 'Yes', 'Maybe'])
phys_health_interview = st.selectbox('phys_health_interview',['Maybe', 'Yes', 'No'])
mental_vs_physical = st.selectbox('mental_vs_physical',['Yes', 'No', "Don't know"])
obs_consequence = st.selectbox('obs_consequence',['No', 'Yes'])
gender = st.selectbox('gender',['female', 'male', 'others'])
df_user = pd.DataFrame([{"Age" : Age,
"Country" : Country,
"self_employed" : self_employed,
"family_history" : family_history,
"treatment" : treatment,
"work_interfere" : work_interfere,
"no_employees" : no_employees,
"remote_work" : remote_work,
"tech_company" : tech_company,
"benefits" : benefits,
"care_options" : care_options,
"wellness_program" : wellness_program,
"anonymity" : anonymity,
"leave" : leave,
"mental_health_consequence" : mental_health_consequence,
"phys_health_consequence" : phys_health_consequence,
"coworkers" : coworkers,
"supervisor" : supervisor,
"mental_health_interview" : mental_health_interview,
"phys_health_interview" : phys_health_interview,
"mental_vs_physical" : mental_vs_physical,
"obs_consequence" : obs_consequence,
"gender" : gender,
}])
if st.button(":rainbow[PREDICT]"):
    st.subheader(":blue[YOUR INPUT]")
    y_predict_user = model_xgb.predict(df_user)
    st.dataframe(df_user)
    if y_predict_user == 0:
        y_predict_user='You have no mental health problem'
    else:
        y_predict_user='you should consult a psychiatrist'
    st.subheader(":blue[PREDICTION]")
    st.markdown(y_predict_user)