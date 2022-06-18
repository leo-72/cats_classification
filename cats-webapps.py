from base64 import encode
from distutils.command.upload import upload
from multiprocessing import dummy
from operator import ge
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import pickle
from sklearn.naive_bayes import GaussianNB

st.write("""
# Klasifikasi Kucing (Web Apps)
Aplikasi berbasis web untuk memprediksi (mengklasifikasi) kucing berdasarkan **breed** kucingnya.
""")

img = Image.open('img/cats.png')
st.image(img, use_column_width=False)

# Setup sidebar
st.sidebar.title('Klasifikasi Kucing')
st.sidebar.subheader('Pilih Parameter')

# Upload File CSV
uploaded_file = st.sidebar.file_uploader("Unggah File CSV", type=["csv"])
if uploaded_file is not None:
    inputan = pd.read_csv(uploaded_file)
else:
    def input_user():
        age = st.sidebar.slider('Umur Kucing', 0, 20, 2)
        gender = st.sidebar.radio('Gender', ('Male', 'Female'))
        size = st.sidebar.selectbox('Size', ('Extra Large', 'Large', 'Medium', 'Small'))
        data = {'age': age,
                'gender': gender,
                'size': size,}
        savingData = pd.DataFrame(data, index=[0])
        return savingData
    inputan = input_user()

# Menggabungkan inputan dan dataset cats
cats_raw = pd.read_csv('cats.csv')
cats = cats_raw.drop(columns=['breed'])
df = pd.concat([inputan, cats], axis=0)

# Encode Data
encode = ['gender', 'size']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df, dummy], axis=1)
    del df[col]

df = df[:1] # mengambil baris pertama(input data user)

# Menampilkan parameter
st.subheader('Parameter Inputan')

if uploaded_file is not None:
    st.write(df)
else:
    st.write('Menunggu file CSV... Saat ini memakai sample default')
    st.write(df)

# Load model NBC
load_model = pickle.load(open('Model/model_cats.pkl', 'rb'))

# Menerapkan NBC
prediksi = load_model.predict(df)
prediksi_proba = load_model.predict_proba(df)

# Menampilkan keterangan label kelas
st.subheader('Keterangan Label Kelas')
breed_cat = np.array(['Abyssinian', 'American Bobtail', 'American Curl'])
st.write(breed_cat)

st.subheader('Hasil Klasifikasi')
st.write(breed_cat[prediksi])

st.subheader('Probabilitas')
st.write(prediksi_proba)
