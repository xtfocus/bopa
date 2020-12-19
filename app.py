from img_classification import predict
from PIL import Image
import streamlit as st
import numpy as np
import keras

uploaded_file = st.file_uploader("Show your fruit (banana, orange, apple or peach only)", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Your pic', use_column_width=True)
    st.write("Classifying. . .")
    label = predict(image)
    st.write("I think this is", label,'!!!')