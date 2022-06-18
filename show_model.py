from base64 import encode
from multiprocessing import dummy
import streamlit as st
import pandas as pd

st.write("""
# Show Model Cats
""")

cats = pd.read_csv('cats.csv')
data = cats.copy()

target = 'breed'
encode = ['gender', 'size']


for col in encode:
    dummy = pd.get_dummies(data[col], prefix=col)
    data = pd.concat([data,dummy], axis=1)
    del data[col] 

data