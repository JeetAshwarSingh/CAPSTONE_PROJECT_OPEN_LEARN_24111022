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

with open('classification_logic.pkl','rb') as f:
    model_logic =pickle.load(f)

with open('classification_xgb.pkl','rb') as f:
    model_xgb =pickle.load(f)

st.title("CLASSIFICATION TASK")
st.header(":blue[MODEL USED FOR CLASSIFICATION]")
st.write('___On this page performance of two models is discussed on the same "cleaned data set"___')
st.write('1. Logistic Regression')
st.write('2. XGBClassifier')
st.write('___Both the models are trained to classify data into two category of people which is to identify what type of people are more prone to suffer from mental health issue and what people are less likely to suffer from it___')
st.header(":blue[LOGISTIC REGRESSION]",divider='blue')
st.write("Logistic regression is the most basic algo which is used to divide data into two category it uses property of sigmoid function to do so")
st.write("Features used:")
st.write("All features except timestamp and state is used")
y_predict = model_logic.predict(x_test)
tup = y_test,y_predict
st.header(':blue[PERFORMANCE OF THE  LOGISTIC REGRESSION]',divider='blue')
st.write("accuracy score of the model :")
st.code(accuracy_score(*tup))
st.write("confusion matrix of the model :")
st.code(confusion_matrix(*tup))
st.write("classification report of the model :")
st.code(classification_report(*tup))
st.write("roc_auc_score of the model is")
st.code(roc_auc_score(*tup))

y_predict = model_xgb.predict(x_test)
tup = y_test,y_predict
st.header(":blue[XGBCLASSIFIER]",divider='blue')
st.write("XGBClassifier is a powerful classifier algorithm which is based on concept of ensemble and boosting")
st.write("Features used:")
st.write("All features except timestamp and state is used")
y_predict = model_xgb.predict(x_test)
tupe = y_test,y_predict
st.header(':blue[PERFORMANCE OF THE XGBCLASSIFIER]',divider='blue')
st.write("accuracy score of the model :")
st.code(accuracy_score(*tupe))
st.write("confusion matrix of the model :")
st.code(confusion_matrix(*tupe))
st.write("classification report of the model :")
st.code(classification_report(*tupe))
st.write("roc_auc_score of the model is")
st.code(roc_auc_score(*tupe))