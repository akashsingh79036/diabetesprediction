import streamlit as st 
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge,Lasso
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler,LabelEncoder
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

uri = "mongodb+srv://akash:akash@cluster0.44hv4.mongodb.net/"
client = MongoClient(uri, server_api=ServerApi('1'))
db=client['diabetes'] # creating a database
# creating a collection
collection=db['diabetes_pred']

def load_model():
    with  open("diabetes_ridge&lasso_final_model.pkl",'rb') as file:
        ridge_model,lasso_model,scaler=pickle.load(file)
    return ridge_model,lasso_model,scaler

def preprocesssing_input_data(data, scaler):
    df = pd.DataFrame([data])
    df_transformed = scaler.transform(df)
    return df_transformed



def predict_data(data):
    ridge_model,lasso_model,scaler = load_model()
    processed_data = preprocesssing_input_data(data,scaler)
    ridge_pred = ridge_model.predict(processed_data)
    lasso_pred = lasso_model.predict(processed_data)
    return ridge_pred,lasso_pred

def main():
    st.title("Diabetes Prediction")
    st.write("enter your data to get a prediction for your diabetes")
    
    age = st.number_input("Age", min_value=0, max_value=100, value=50)
    sex = st.number_input("Sex (0 for female, 1 for male)", min_value=0, max_value=1, value=0)
    bmi = st.number_input("BMI", min_value=0, max_value=50, value=25)
    bp = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=200, value=120)
    s1 = st.number_input("s1 (tc)", min_value=0, max_value=15, value=5)
    s2 = st.number_input("s2 (ldl)", min_value=0, max_value=10, value=3)
    s3 = st.number_input("s3 (hdl)", min_value=0, max_value=10, value=4)
    s4 = st.number_input("s4 (tch)", min_value=0, max_value=15, value=5)
    s5 = st.number_input("s5 (ltg)", min_value=0, max_value=10, value=3)
    s6 = st.number_input("s6 (glu)", min_value=0, max_value=200, value=100)
    
    if st.button('predict_diabetes'):
        user_data= {
            "age":age,
            "sex":sex,
            "bmi":bmi,
            "bp":bp,
            "s1":s1,
            "s2":s2,	
            "s3":s3,
            "s4":s4,
            "s5":s5,
            "s6":s6
        }
        data = [user_data["age"], user_data["sex"], user_data["bmi"], user_data["bp"],
                user_data["s1"], user_data["s2"], user_data["s3"], user_data["s4"],
                user_data["s5"], user_data["s6"]]
        ridge_pred,lasso_pred = predict_data(data)
        st.success(f"your prediction result is {ridge_pred}")
        st.success(f"your prediction result is {lasso_pred}")
        collection.insert_one(user_data)
        #convert the data to the python compatible format of data type
        
if __name__ == "__main__":
    main()