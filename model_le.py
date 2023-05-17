import streamlit as st
import pickle
import numpy as np


def load_model():
    with open('C:/Users/v-kas/OneDrive/Desktop/MachineLearning_project/Cropdamage_model_xg.pkl', 'rb') as file:
        data_load = pickle.load(file)
    return data_load


data = load_model()

clf_loaded_model = data["model"]
# test_le_Crop_Type = data["test_le_Crop_Type"]
test_le_Soil_Type = data["test_le_Soil_Type"]
test_le_Pesticide_Use_Category = data["test_le_Pesticide_Use_Category"]
test_le_Season = data["test_le_Season"]


def show_predict_page():
    # Crop_Type_de = ("Kharif", "Rabi")
    Soil_Type_de = ("Alluvial", "Black-Cotton")
    Season_de = ("Summer", "Monsoon", "Winter")
    Pesticide_Use_Category_de = ("Insecticides", "Bactericides", "Herbicides")

    st.title('Crop Damage Prediction')
    Estimated_Insects_Count = st.text_input('Count of insects')
    # Crop_Type = st.selectbox('Crop_Type details', Crop_Type_de)
    Soil_Type = st.selectbox('Soil_Type details', Soil_Type_de)
    Pesticide_Use_Category = st.selectbox('Pesticide_Use_Category details', Pesticide_Use_Category_de)
    Number_Doses_Week = st.text_input('Number_Doses_Week details')
    Number_Weeks_Used = st.text_input('Number_Weeks_Used details')
    Number_Weeks_Quit = st.text_input('Number_Weeks_Quit details')
    Season = st.selectbox('Season details', Season_de)

    ok = st.button("Crop Damage Prediction")
    if ok:
        x = np.array([[Estimated_Insects_Count, Soil_Type, Pesticide_Use_Category, Number_Doses_Week,
                      Number_Weeks_Used, Number_Weeks_Quit, Season]])
        x = np.array([[Estimated_Insects_Count, Soil_Type, Pesticide_Use_Category, Number_Doses_Week,
                       Number_Weeks_Used, Number_Weeks_Quit, Season]])

        # Check if all fields are filled
        if '' in x:
            st.warning("Please fill all details")
            return
        # x[:, 1] = le_Crop_Type.transform(x[:, 1])
        x[:, 1] = test_le_Soil_Type.transform(x[:, 1])
        x[:, 6] = test_le_Season.transform(x[:, 6])
        x[:, 2] = test_le_Pesticide_Use_Category.transform(x[:, 2])
        x = x.astype(float)

        predict_crop = clf_loaded_model.predict(x)
        print('values', predict_crop)

        if predict_crop == 0:
            st.subheader('The crop damage prediction will be: Minimal Damage')
        elif predict_crop == 1:
            st.subheader('The crop damage prediction will be: Partial Damage')
        else:
            st.subheader('The crop damage prediction will be: Significant Damage')
