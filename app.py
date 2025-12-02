import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os

# Hugging Face specific setup
st.set_page_config(page_title="Aerial AI", layout="wide")

@st.cache_resource
def load_model():
    """Load model for Hugging Face"""
    try:
        model = tf.keras.models.load_model('models/final_model.h5')
        return model
    except:
        return None

def main():
    st.title("🛸 Aerial Object Classification")
    model = load_model()
    
    uploaded_file = st.file_uploader("Upload image", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file and model:
        image = Image.open(uploaded_file)
        st.image(image, width=300)
        
        # Process and predict
        img = image.resize((150, 150))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        prediction = model.predict(img_array, verbose=0)
        score = float(prediction[0][0])
        
        if score < 0.5:
            st.success(f"🐦 BIRD (Confidence: {(1-score)*100:.1f}%)")
        else:
            st.success(f"🚁 DRONE (Confidence: {score*100:.1f}%)")

if __name__ == "__main__":
    main()