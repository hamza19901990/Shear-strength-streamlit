import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import streamlit as st
from PIL import Image

st.write("""
# Shear Strength of Reinforced Concrete Beams
This app predicts the **Shear strength of reinforced concrete beams using machine learning**!
Data obtained from the [Zhang T, Visintin P, Oehlers DJ (2015) Shear strength of RC
beams with steel stirrups. J Struct Eng 142(2):04015135](https://doi.org/10.1061/(ASCE)ST.1943-541X.0001404).
***
""")
st.write('---')
image=Image.open(r'hqdefault.jpg')
st.image(image, use_column_width=True)
req_col_names = ["b(mm)", "d(mm)", "Cs Conc.(Mpa)", "(As/bd)%", "Long. steel yielding (Mpa)",
                 "trans. reinforc. ratio %", " trans. reinf. (Mpa)", "Vexp (KN)", ]
def get_input_features():
    bmm = st.sidebar.slider('b(mm))', 76,457,100)
    dmm = st.sidebar.slider('d(mm)', 95,851,200)
    Cs = st.sidebar.slider('Cs Conc.(Mpa)',13,51,20)
    Assteel = st.sidebar.slider('(As/bd)%', 1,5,2)
    Long_steel_yielding_Mpa = st.sidebar.slider('Long. steel yielding (Mpa))', 300,707,400)
    trans_reinforc_ratio = st.sidebar.slider('trans. reinforc. ratio %', 0.1,1.9,0.2)
    trans_reinf_yielding_Mpa = st.sidebar.slider('trans. reinf. yielding (Mpa)', 159,820,200)
    FineAggregare = st.sidebar.slider('Vexp (KN)', 594,992,800)

    data_user = {'b(mm))': bmm,
            'd(mm)': dmm,
            'Cs Conc.(Mpa)': Cs,
            '(As/bd)%': Assteel,
            'Long. steel yielding (Mpa))': Long_steel_yielding_Mpa,
            'trans. reinforc. ratio %': trans_reinforc_ratio,
            'trans. reinf. yielding (Mpa)': trans_reinf_yielding_Mpa}
    features = pd.DataFrame(data_user, index=[0])
    return features

df = get_input_features()
# Main Panel

# Print specified input parameters
st.header('Specified Input parameters')
st.write(df)
st.write('---')

# Reads in saved classification model
import pickle
load_clf = pickle.load(open('shear.pkl', 'rb'))
st.header('Prediction of Shear Force (Kn)')

# Apply model to make predictions
prediction = load_clf.predict(df)
st.write(prediction)
st.write('---')
st.header('Predicted vs Measured Shear Force (Kn) with R^2=0.9')
image3=Image.open(r'Figure_2.png')
st.image(image3, use_column_width=True)
st.header('Feature Importance')
image2=Image.open(r'Figure_1.png')
st.image(image2, use_column_width=True)