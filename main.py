import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import gdown
import os

st.set_page_config(page_title="Identifikasi & Performa Model", layout="centered")

MODEL_ID = "1XIdB6g1TnY4OmFbY2XbkwGawP7a-tzyr"   # Ganti dengan file ID Google Drive kamu
MODEL_PATH = "model_inceptionv3.keras"

@st.cache_resource
def load_model():
    # Cek apakah model sudah ada
    if not os.path.exists(MODEL_PATH):
        # Link Google Drive direct download via gdown
        url = f"https://drive.google.com/uc?id={MODEL_ID}"
        with st.spinner("Mengunduh model dari Google Drive..."):
            gdown.download(url, MODEL_PATH, quiet=False)
    # Load model dari lokal
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()
class_names = [
    "Belimbing Wuluh", "Daun Jeruk", "Daun Kari", "Daun Katuk", "Daun Kelor",
    "Daun Kemangi", "Daun Kunyit", "Daun Sirih", "Daun Sirsak", "Jambu Biji"
]

st.sidebar.title("Navigasi")
menu = st.sidebar.radio(
    "Pilih halaman",
    ("Identifikasi", "Performa Model")
)

# Kunci unik untuk file_uploader supaya bisa direset
if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 0

# ------------------- IDENTIFIKASI ----------------------
if menu == "Identifikasi":
    st.title("Identifikasi Gambar dengan InceptionV3")
    st.write("Upload gambar dan klik **Identifikasi**.")

    if "uploaded_img" not in st.session_state:
        st.session_state["uploaded_img"] = None

    uploaded_file = st.file_uploader(
        "Pilih gambar...", 
        type=["jpg", "jpeg", "png"],
        key=st.session_state["uploader_key"]
    )

    # Saat file baru diupload, simpan ke session_state
    if uploaded_file is not None:
        st.session_state["uploaded_img"] = uploaded_file

    # Jika sudah ada gambar, tampilkan
    if st.session_state["uploaded_img"] is not None:
        image = Image.open(st.session_state["uploaded_img"]).convert("RGB")
        st.image(image, caption="Gambar yang di-upload", use_container_width=True)
        img_resized = image.resize((299, 299))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        if st.button("Hapus Gambar"):
            st.session_state["uploaded_img"] = None
            st.session_state["uploader_key"] += 1  # Reset file_uploader
            st.rerun()

        if st.button("Identifikasi"):
            with st.spinner("Memproses..."):
                pred = model.predict(img_array)
                predicted_class = class_names[np.argmax(pred)]
                st.success(f"Hasil Identifikasi: **{predicted_class}**")
    else:
        st.write("Belum ada gambar yang diupload.")

# ------------- PERFORMA MODEL -------------------
elif menu == "Performa Model":
    st.title("Performa Model")
    st.write("Visualisasi confusion matrix dan classification report dari hasil evaluasi model.")

    # Menampilkan gambar confusion matrix dari file lokal
    st.subheader("Confusion Matrix")
    try:
        st.image("conf_matrix.png", caption="Confusion Matrix", use_container_width=True)
    except Exception as e:
        st.warning("File 'conf_matrix.png' tidak ditemukan di folder. Silakan cek nama file dan lokasi.")

    # Menampilkan gambar classification report dari file lokal
    st.subheader("Classification Report")
    try:
        st.image("classification_report.png", caption="Classification Report", use_container_width=True)
    except Exception as e:
        st.warning("File 'classification_report.png' tidak ditemukan di folder. Silakan cek nama file dan lokasi.")
