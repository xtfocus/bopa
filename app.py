from img_classification import predict
from PIL import Image
import streamlit as st
from numpy import array


uploaded_file = st.file_uploader("Show your fruit: apple, avocado, banana, cherry, grapes, kiwi, mango, orange, peach, pear, pineapple, strawberry or watermelon", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Your pic', use_column_width=True)
    st.write("Classifying. . .")
    label = predict(image)
    st.write("I think this is", label,'!!!')