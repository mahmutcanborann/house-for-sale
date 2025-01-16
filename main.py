from sklearn.linear_model import LinearRegression
import streamlit as st
import os
import base64
import numpy as np
import pandas as pd

# CSV dosyasını okuma
df = pd.read_csv(r"C:\Users\mahmu\Downloads\data (1).csv")

# Statik dosya yolu
image_path = r"C:\Users\mahmu\OneDrive\Desktop\x.jpg"

# Resmi base64 formatında encode etme
with open(image_path, "rb") as image_file:
    encoded_image = base64.b64encode(image_file.read()).decode()

# CSS ile arka plan resmi ayarlama
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpeg;base64,{encoded_image}");
        background-size: cover;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Veri hazırlama
df = df[['price', 'bedrooms', 'bathrooms', 'yr_built']]
df = pd.get_dummies(df, drop_first=True)
y = df['price']
x = df.drop('price', axis=1)

# Model oluşturma ve eğitme
lr = LinearRegression()
model = lr.fit(x, y)

st.title("Ev Bilgisi Girişi")

# Kullanıcıdan giriş değerlerini alma
oda_sayisi = st.number_input("Oda Sayısı", min_value=1, max_value=10, value=1, step=1)
banyo_sayisi = st.number_input("Banyo Sayısı", min_value=1, max_value=5, value=1, step=1)
bina_yasi = st.number_input("Bina Yaşı", min_value=0, max_value=100, value=0, step=1)

# Buton ekleme ve tahmin yapma
if st.button("Göster"):
    # Kullanıcıdan alınan değerleri modele uygun formatta veriyoruz
    input_data = np.array([[oda_sayisi, banyo_sayisi, bina_yasi]])
    prediction = model.predict(input_data)
    st.write(f"Tahmini Fiyat: {prediction[0]:,.2f} TL")
