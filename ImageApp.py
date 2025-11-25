import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2

model = tf.keras.models.load_model("model.h5")

st.set_page_config(page_title="ImageApp", layout="wide")

CUSTOM_CSS = r"""
    <style>
:root[data-theme="light"] {
  --bg: #878789;
  --card: rgba(255,255,255,0.06);
  --text: #0b1220;
  --accent1: linear-gradient(90deg,#7c3aed, #06b6d4);
}
:root[data-theme="dark"] {
  --bg: #0b0b13;
  --card: rgba(255,255,255,0.04);
  --text: #dbeafe;
  --accent1: linear-gradient(90deg,#06b6d4, #7c3aed);
}

/* Apply glass card effect to streamlit elements */
main .block-container {
  background: linear-gradient(180deg, rgba(255,255,255,0.01), rgba(255,255,255,0.00));
  padding: 1.6rem 2rem;
}
section[data-testid="stSidebar"] .css-1d391kg {
  background: transparent;
}
.css-1d391kg, .css-1d391kg .stButton button {
  border-radius: 14px;
}

/* Title style */
.header {
  display:flex; align-items:center; gap:12px;
}
.logo-circle {
  width:56px;height:56px;border-radius:12px;
  background: var(--accent1);
  display:flex;align-items:center;justify-content:center;color:white;font-weight:700;
  box-shadow: 0 8px 30px rgba(99,102,241,0.15);
}

/* Card style used in columns */
.card {
  background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
  border: 1px solid rgba(255,255,255,0.04);
  padding: 16px;
  border-radius: 12px;
}

/* small pills */
.pill {
  display:inline-block;padding:6px 10px;border-radius:999px;font-size:12px;background:rgba(255,255,255,0.03);
}

/* bot bubble */
.bot {
  background: linear-gradient(90deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
  padding: 10px 12px;border-radius:12px;margin:6px 0;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
with st.sidebar:
    st.markdown(
        "<div style='display:flex;align-items:center;gap:10px'><div class='logo-circle'>IApp</div><div><h3 style='margin:0'>ImageApp</h3><div style='font-size:12px;color:gray'>TensorFlow · Sklearn · Explainability</div></div></div>",
        unsafe_allow_html=True)
    st.markdown("------")
    st.markdown("**Quick Settings**")
    st.markdown("-------")

st.write("Know which Image you Uploaded")
st.markdown("--------")
st.write(
    "Try uploading an image to see the name of the Image. This code is open source and available [here](https://github.com/Brian342/ImageClassification) on GitHub"
)
st.markdown("---------")
col1, col2 = st.columns([2, 1])

uploaded = st.file_uploader("Upload an Image here", type=['png', 'jpeg', 'jpg'])

if uploaded:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # on the main page

    with col1:
        st.image(img_rgb, use_column_width=True)
    img_resized = cv2.resize(img_rgb, (32, 32))
    img_input = np.expand_dims(img_resized / 255.0, axis=0)

    # model predicts
    predict = model.predict(img_input)
    predict_class = np.argmax(predict)

    with col2:
        st.write("Image Name :gear:")
        st.success(f"Prediction: **{predict_class}**")
else:
    with col1:
        st.info("Unload an image to continue")


