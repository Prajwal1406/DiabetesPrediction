import streamlit as st
import numpy as np
import pickle
page_bg_img = '''
<style>
body {
background-image: url("https://images.unsplash.com/photo-1542281286-9e0a16bb7366");
background-size: cover;
}
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)
classifier=pickle.load(open('classifier.pkl','rb'))
scalar=pickle.load(open('scalar.pkl','rb'))
def input_data(Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age):
    input_data = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
    input_np = np.array(input_data)
    input_reshaped = input_np.reshape(1, -1)
    st_data = scalar.transform(input_reshaped)
    prediction = classifier.predict(st_data)
    return prediction
    
st.title('Diabetes Prediction System')
col1, col2,col3 = st.columns(3)
with col1:
    Pregnancies=st.number_input('Pregnancies')
with col2:
    Glucose=st.number_input('Glucose')
with col3:
    BloodPressure=st.number_input('BloodPressure')
with col1:
    SkinThickness=st.number_input('SkinThickness')
with col2:
    Insulin=st.number_input('Insulin')
with col3:
    BMI=st.number_input('BMI')
with col1:
    DiabetesPedigreeFunction=st.number_input('DiabetesPedigreeFunction')
with col2:
    Age=st.number_input('Age')

if st.button("diagonise"):
    
    diagonise = input_data(Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age)
    if (diagonise == 1):
        diag = 'Is diabetic'
    else:
        diag = 'Non diabetic'
    st.success(diag)