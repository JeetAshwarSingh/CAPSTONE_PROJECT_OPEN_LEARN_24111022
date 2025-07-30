import streamlit as st
import pickle 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_absolute_error,root_mean_squared_error

df = pd.read_csv('cleaned_survey.csv')

st.title("CAPSTONE PROJECT")
st.header(":blue[ABOUT THIS PROJECT]",divider='blue')
st.write('___Greetings to everyone reading this. This website is created to showcase my Capstone project, which I completed on 28 July in the year 2025.___')
st.write("___In this website, a project related to various machine learning models built on many different machine learning algorithms, like  linear regression, XGBOOST, Logistic regression and many more, with these their are multi insights along with EDA___")
st.header(":blue[ABOUT DATASET]",divider='blue')
st.write("___This dataset is taken from the ‘2014 Mental Health in Tech Survey’ conducted by ‘Open Sourcing Mental Illness (OSMI)’. The purpose of this dataset is to evaluate working class people’s mental health status. dataset initially contained a total 1259 rows along with 27 columns___")
st.write('various features of this dataset are')
st.dataframe(df.columns.to_list())

st.header(":blue[ABOUT OPEN LEARN COHORT]",divider='blue')
st.write('[Open learn cohort](https://www.openlearn.org.in/) was organised at Dr. B. R. Ambedkar National Institute of Technology, Jalandhar in year of 2025 Aimed to bring an AI/ML revolution in the college. Open cohort is led by key industry figures and pioneers are mentored by experts in AI/ML domain')