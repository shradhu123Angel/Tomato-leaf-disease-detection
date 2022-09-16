import streamlit as st
from PIL import Image
import numpy as np
import cv2

st.title("Leaf disease classification system")
st.header("Tomato leaf disease Classification")
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    label_names = ['Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']
    label_names.sort()
    net = cv2.dnn.readNetFromONNX('converted_model.onnx')
    img = cv2.resize(np.array(image),(224,224))
    img = np.array([img]).astype('float64') / 255.0
    net.setInput(img)
    out = net.forward()
    index = np.argmax(out[0])
    label =  label_names[index].capitalize()
    st.write("Prediction: ", label)