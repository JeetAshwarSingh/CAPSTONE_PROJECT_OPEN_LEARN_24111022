import streamlit as st
import pickle 
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score,mean_absolute_error,root_mean_squared_error


df = pd.read_csv('cleaned_survey.csv')

x = df.drop(columns=['Age'])
y = df['Age']
x_train,x_test,y_train,y_test = train_test_split( x , y , train_size=0.8,random_state=42 )

with open('regression_task_linear.pkl','rb') as f:
    model_regression =pickle.load(f)

with open('regression_task_xgb.pkl','rb') as f:
    model_xgb =pickle.load(f)

st.title("REGRESSION TASK")
st.header(":blue[MODELS FOR REGRESSION]",divider='blue')
st.write('___On this page performance of two models is discussed on the same "cleaned data set"___')
st.write('1. Linear Regression')
st.write('2. XGBRegressor')
st.write('___Although data is not suitable for regression and fits better for classification but for the sake of understanding and clarity, regression is also applied on the data.___')
st.write('___both model predict same target feature “Age” that is which Age group of people are more prone towards depression___')
st.header(":blue[LINEAR REGRESSION]",divider='blue')
st.write("Linear regression is the most basic and fundamental model in machine learning it is the foundation block of machine learning")
st.write("Features used:")
st.write("All features except timestamp and state is used")
y_predict = model_regression.predict(x_test)
tup=y_test,y_predict
st.header(':blue[PERFORMANCE OF THE  REGRESSION MODEL]',divider='blue')
st.write("R_2 score of linear regression model is")
st.code(r2_score(*tup))
st.write("MAE of model is")
st.code(mean_absolute_error(*tup))
st.write("RMSE of model is")
st.code(root_mean_squared_error(*tup))
st.subheader(":grey['Y_TEST PLOT'  vs   'Y_PREDICT plot']",divider='grey')
img,ax =plt.subplots()
plt.scatter( y=y_test,x = y_predict,color = 'blue')
plt.scatter( y=y_test,x = y_test,color = 'yellow')
plt.xlabel("Y_PREDICT")
plt.ylabel("Y_TEST PLOT")
st.pyplot(img)
     
st.header(":blue[XGBOOST]",divider='blue')
st.write("XGBoost is a boosting based advanced algorithm which works with trees to reach to reach its desired conclusions")
st.write("Features used:")
st.write("All features except timestamp and state is used")
y_predict = model_xgb.predict(x_test)
tup=y_test,y_predict
st.header(':blue[PERFORMANCE OF THE  XGBOOST]',divider='blue')
st.write("R_2 score of XGBOOST model is")
st.code(r2_score(*tup))
st.write("MAE of model is")
st.code(mean_absolute_error(*tup))
st.write("RMSE of model is")
st.code(root_mean_squared_error(*tup))
st.subheader(":grey['Y_TEST PLOT'  vs   'Y_PREDICT plot']",divider='grey')
img,ax =plt.subplots()
plt.scatter( y=y_test,x = y_predict,color = 'green')
plt.scatter( y=y_test,x = y_test,color = 'red')
plt.xlabel("Y_PREDICT")
plt.ylabel("Y_TEST PLOT")
st.pyplot(img)
col1, col2, col3 = st.columns([100, 99, 100])
st.write(':red[since data is not good for fitting on regressor thats why it performed so bad on regressors]')