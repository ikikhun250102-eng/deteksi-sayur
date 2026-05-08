import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# Konfigurasi Halaman
st.set_page_config(page_title="Deteksi Tanaman", page_icon="🌶️")

# 1. Load Model
@st.cache_resource
def load_my_model():
    # Pastikan nama file ini sama dengan file .h5 kamu
    model = tf.keras.models.load_model('model_sayur.h5')
    return model

model = load_my_model()

# 2. Label Kelas
class_names = ['Cabai', 'Terong', 'Tomat']

st.title("🍅 Klasifikasi Cabai, Terong, & Tomat")
st.write("Unggah foto sayuran di bawah ini untuk dideteksi oleh AI.")

# 3. Upload File
file = st.file_uploader("Pilih gambar...", type=["jpg", "png", "jpeg"])

def import_and_predict(image_data, model):
    size = (224, 224)    
    image = ImageOps.fit(image_data, size, Image.LANCZOS)
    img_array = np.asarray(image)
    img_reshape = img_array[np.newaxis, ...] # Menambah dimensi batch
    img_reshape = img_reshape / 255.0 # Normalisasi
    
    prediction = model.predict(img_reshape)
    return prediction

if file is None:
    st.text("Silakan unggah file gambar")
else:
    image = Image.open(file)
    st.image(image, use_container_width=True)
    
    # Prediksi
    predictions = import_and_predict(image, model)
    
    label = class_names[np.argmax(predictions)]
    confidence = np.max(predictions) * 100
    
    # Tampilkan Hasil
    st.subheader(f"Hasil Prediksi: {label}")
    st.info(f"Tingkat Keyakinan: {confidence:.2f}%")
    
    # Tambahkan Rekomendasi
    if label == 'Cabai':
        st.write("💡 **Tips:** Simpan cabai di wadah kering agar tidak cepat busuk.")
    elif label == 'Terong':
        st.write("💡 **Tips:** Terong segar biasanya memiliki kulit yang kencang dan mengkilap.")
    else:
        st.write("💡 **Tips:** Tomat kaya akan Vitamin C dan baik untuk kesehatan kulit.")