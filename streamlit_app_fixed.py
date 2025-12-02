import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import os

st.set_page_config(page_title="Aerial Object Classification", layout="wide")

class FixedAerialClassifier:
    def __init__(self):
        self.model = None
        self.load_model()
    
    def load_model(self):
        model_path = "models/best_model.h5"
        if os.path.exists(model_path):
            try:
                self.model = keras.models.load_model(model_path)
                st.sidebar.success("✅ Model loaded!")
            except Exception as e:
                st.sidebar.error(f"❌ Error: {e}")
        else:
            st.sidebar.error("❌ Model not found!")
    
    def preprocess_image(self, image):
        image = image.resize((150, 150))
        img_array = np.array(image) / 255.0
        if len(img_array.shape) == 3 and img_array.shape[-1] == 4:
            img_array = img_array[:, :, :3]
        return np.expand_dims(img_array, axis=0)
    
    def predict(self, image):
        if self.model is None:
            return None, None
        
        try:
            processed_image = self.preprocess_image(image)
            prediction = self.model.predict(processed_image, verbose=0)
            confidence = float(prediction[0][0])
            
            # 🎯 FIXED: Class labels were inverted in training
            # Now: score < 0.5 = DRONE, score > 0.5 = BIRD
            if confidence > 0.5:
                class_name = "Bird"
                final_confidence = confidence
            else:
                class_name = "Drone"
                final_confidence = 1 - confidence
            
            return class_name, final_confidence
            
        except Exception as e:
            st.error(f"Error: {e}")
            return None, None

def main():
    classifier = FixedAerialClassifier()
    
    st.title("🛸 Aerial Object Classification - FIXED")
    st.write("Classify images as **Bird** or **Drone**")
    
    # Model info
    st.sidebar.info("""
    **Model Info:**
    - Accuracy: 82.79%
    - Fixed Class Labels
    - Image Size: 150x150
    """)
    
    uploaded_file = st.file_uploader("Upload image...", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Uploaded Image")
            try:
                image = Image.open(uploaded_file)
                st.image(image, use_column_width=True)
                st.info(f"Image size: {image.size}")
            except Exception as e:
                st.error(f"Error loading image: {e}")
                return
            
        with col2:
            st.subheader("Prediction Results")
            
            with st.spinner("🔍 Analyzing image..."):
                class_name, confidence = classifier.predict(image)
            
            if class_name:
                confidence_percent = confidence * 100
                
                # Display with color coding
                if class_name == "Bird":
                    st.success(f"## 🐦 Prediction: BIRD")
                    st.info("**Characteristics:** Organic shapes, feathers, natural contours")
                else:
                    st.warning(f"## 🚁 Prediction: DRONE") 
                    st.info("**Characteristics:** Mechanical structure, propellers, angular shapes")
                
                # Confidence indicator
                if confidence_percent >= 80:
                    confidence_color = "🟢"
                elif confidence_percent >= 60:
                    confidence_color = "🟡" 
                else:
                    confidence_color = "🔴"
                
                st.write(f"### {confidence_color} Confidence: {confidence_percent:.2f}%")
                
                # Debug info
                with st.expander("Technical Details"):
                    st.write(f"Model: best_model.h5")
                    st.write(f"Input shape: 150x150x3")
                    st.write(f"Class mapping: <0.5 = Drone, >0.5 = Bird")
                    
            else:
                st.error("Prediction failed!")

if __name__ == "__main__":
    main()