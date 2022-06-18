from base64 import encode
from copyreg import pickle
from multiprocessing import dummy
from sklearn.naive_bayes import GaussianNB
import streamlit as st
import pandas as pd
import pickle

st.write("""
# Create Model Cats
""")

cats = pd.read_csv('cats.csv')
data = cats.copy()

target = 'breed'
encode = ['gender', 'size']


for col in encode:
    dummy = pd.get_dummies(data[col], prefix=col)
    data = pd.concat([data,dummy], axis=1)
    del data[col] 

target_mapper = {'Abyssinian':0, 'American Bobtail':1, 'American Curl':2,}

def target_encode(val):
    return target_mapper[val]

data['breed'] = data['breed'].apply(target_encode)

X = data.drop(['breed'], axis=1)
y = data['breed']

# Model NBC
model = GaussianNB()
model.fit(X, y)

# Menyimpan Model   
pickle.dump(model, open('model_cats.pkl', 'wb'))
st.write('Model berhasil disimpan')