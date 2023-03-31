import numpy as np
import pickle
import pandas as pd
import streamlit as st

pickle_in = open("model.pkl","rb")
regressor = pickle.load(pickle_in)

def predict_house_price(features_value):
    return regressor.predict(features_value)


st.title("House Sale Prediction")
OverallQual = st.text_input("Quality","Type here")
YearBuilt = st.text_input("year built","Type here")
YearRemodAdd = st.text_input("Remodel date","Type here")
TotalBsmtSF = st.text_input("TotalBsmtSF","Type here")
FirstFlrSF = st.text_input("1stFlrSF","Type here")
GrLivArea = st.text_input("GrLivArea","Type here")
FullBath = st.text_input("FullBath","Type here")
TotRmsAbvGrd = st.text_input("TotRmsAbvGrd","Type here")
GarageCars = st.text_input("GarageCars","Type here")
GarageArea = st.text_input("GarageArea","Type here")
MSSubClass = st.text_input("MSSubClass","Type here")
MSZoning = st.text_input("MSZoning","Type here")
Neighborhood = st.text_input("Neighborhood","Type here")
Condition1 = st.text_input("Condition1","Type here")

data = [[OverallQual,YearBuilt,YearRemodAdd, TotalBsmtSF,FirstFlrSF,GrLivArea,FullBath,TotRmsAbvGrd,GarageCars,GarageArea,MSSubClass,MSZoning,Neighborhood,Condition1]]
result =""
if st.button("Predict"):
    result = predict_house_price(data)
st.success('The sale price is{}'.format(result))

