import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle as pkl
import sklearn

@st.cache_data()  # Menyimpan model di cache untuk performa
def load_classification_model():
    with open('model_salary_rf.pkl', 'rb') as file:
        model = pkl.load(file)
    
    with open('scaler.pkl', 'rb') as scale:
        scaler = pkl.load(scale)
    return model, scaler

st.title('Prediksi Gaji Karyawan dengan Random Forest')
st.header('Salary Predictor')
st.write('Projek ini bertujuan sebagai media pembelajaran Machine Learning')

model, scaler = load_classification_model()
# Input usia
usia = st.number_input('Usia', value=0)

# Input jumlah proyek yang telah diselesaikan
jml_proyek = st.number_input('Jumlah proyek', value=0)

# Menyimpan input dalam array
data_input = [usia, jml_proyek]

# Scaling data input
scaled_input = scaler.transform([data_input])

if st.button('Prediksi'):
    if data_input:
        prediction = model.predict(scaled_input)
        st.write(f'Prediksi: {prediction}')
    else:
        st.write('Silakan masukkan teks.')


