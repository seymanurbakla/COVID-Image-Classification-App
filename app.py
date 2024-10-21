#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 16:23:29 2024

@author: seynoma
"""

import streamlit as st
import numpy as np
import PIL.Image as img
import pickle
import matplotlib.pyplot as plt

# Modelleri yükleme
@st.cache(allow_output_mutation=True)
def load_models():
    rf = pickle.load(open('rf.pkl', 'rb'))
    lr = pickle.load(open('lr.pkl', 'rb'))
    svm = pickle.load(open('svm.pkl', 'rb'))
    dt = pickle.load(open('dt.pkl', 'rb'))
    return rf, lr, svm, dt

# Model açıklamaları
model_descriptions = {
    "Random Forest": "Random Forest, birçok karar ağacı kullanarak sınıflandırma yapar ve genellikle yüksek doğruluk sağlar.",
    "Logistic Regression": "Logistic Regression, iki sınıflı bir sonuç tahmin etmek için kullanılan istatistiksel bir yöntemdir.",
    "SVM": "Destek Vektör Makineleri (SVM), veriyi sınıflandırmak için en iyi hiper düzlemi bulmaya çalışır.",
    "Decision Tree": "Karar ağaçları, veriyi ağaç yapısında bölerek karar vermeyi kolaylaştırır."
}

# Uygulamanın başlatılması
st.title("COVID Tespit Uygulaması")

# Model seçimi
model_option = st.selectbox("Kullanmak istediğiniz modeli seçin:", list(model_descriptions.keys()))
st.write(model_descriptions[model_option])  # Model açıklamasını göster

# Performans verileri
model_performances = {
    "Random Forest": 0.95,
    "Logistic Regression": 0.89,
    "SVM": 0.92,
    "Decision Tree": 0.85
}

# Performans grafiği oluşturma
st.write("Modellerin geçmiş performansları:")
plt.figure(figsize=(8, 4))
plt.bar(model_performances.keys(), model_performances.values(), color='blue')
plt.xlabel("Modeller")
plt.ylabel("Doğruluk Oranı")
plt.ylim(0, 1)
plt.title("Model Performansı")
st.pyplot(plt)

# Dosya yükleme
uploaded_file = st.file_uploader("Lütfen bir COVID görüntüsü yükleyin", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Yüklenen resmi açma ve işleme
    image = img.open(uploaded_file).convert("L")  # Gri tonlamaya çeviriyoruz
    image = image.resize((28, 28))  # 28x28 boyutlandırma
    img_array = np.array(image).flatten()  # Resmi düzleştirme
    img_array = np.reshape(img_array, (1, -1))  # Modelin beklediği boyuta getiriyoruz

    # Modelleri yükle
    rf, lr, svm, dt = load_models()

    # Tahmin yapma
    if model_option == "Random Forest":
        prediction = rf.predict(img_array)
    elif model_option == "Logistic Regression":
        prediction = lr.predict(img_array)
    elif model_option == "SVM":
        prediction = svm.predict(img_array)
    elif model_option == "Decision Tree":
        prediction = dt.predict(img_array)

    # Tahmin sonucunu göster
    st.write(f"Tahmin Sonucu: {'COVID-19 Pozitif' if prediction[0] == 1 else 'COVID-19 Negatif'}")
