# Liberary and modules
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sklearn
import bz2
import joblib
import gzip


        
# All features user input.
st.title("Zomato Bangalore Restaurants")
name = st.text_input("name","Type here")
online_order = st.text_input("online_order","Type here")
book_table = st.text_input("book_table","Type here")
votes = st.text_input("votes","Type here")
location = st.text_input("location","Type here")
rest_type = st.text_input("rest_type","Type here")
cuisines = st.text_input("cuisines","Type here")
Cost_for_couple = st.text_input("Cost_for_couple","Type here")
Type = st.text_input("Type","Type here")
city = st.text_input("city","Type here")



    
    
 # pandas dataframe format   
df = pd.DataFrame({"name":[name],"online_order":[online_order],"book_table":[book_table],"votes":[votes],"location":[location],
                   "rest_type":[rest_type],"cuisines":[cuisines],"Cost_for_couple":[Cost_for_couple],"Type":[Type],"city":[city]})   
    



def encode_categorical_columns(df):
    categorical_columns = df.loc[:, df.dtypes == 'object'].columns
    for col in categorical_columns:
        df[col] = pd.factorize(df[col])[0]
    return df
encode_categorical_columns(df)


with gzip.open("C:/Users/145568/Downloads/Practice/Machine_Learning/EP_T/Project_1/model.pkl.gz", "rb") as file:
    #Load model
    load_model = pickle.load(file)



#Prediction results based on input data from user.
prediction = load_model.predict(df)
if st.button("Input Result"):
    if(prediction>=1):
        st.success("restaurant has a good rate and could by highly recommended according to customer reviews")
    else:
        st.error("restaurant does not get a high rate")