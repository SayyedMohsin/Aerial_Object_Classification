import streamlit as st
import numpy as np
from PIL import Image
import os
from ensemble_model import EnsembleClassifier

st.set_page_config(page_title="Aerial Classification - ENSEMBLE", layout="wide")

def main():
    st.title("🎯 Advanced Aerial Classification - ENSEMBLE")
    st.write("**Combining Multiple AI Models for Maximum Accuracy**")
    
    classifier = EnsembleClassifier()
    
    if not classifier.models:
        st.warning("⚠️ Some models missing. Using available models.")
    
    uploaded_file = st.file_uploader("Upload difficult drone/bird image", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📤 Image")
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True)
            st.info(f"Size: {image.size} | Mode: {image.mode}")
        
        with col2:
            st.subheader("🎯 Ensemble Prediction")
            
            with st.spinner("🤖 Multiple AI models analyzing..."):
                class_name, confidence = classifier.predict_ensemble(image)
            
            if class_name:
                conf_percent = confidence * 100
                
                if class_name == "BIRD":
                    st.success(f"## 🐦 ENSEMBLE: BIRD")
                else:
                    st.success(f"## 🚁 ENSEMBLE: DRONE")
                
                st.write(f"**Confidence: {conf_percent:.1f}%**")
                
                # Model details
                st.info("**Models Used:**")
                for model_name in classifier.models.keys():
                    st.write(f"- ✅ {model_name.upper()} Model")
                
                if conf_percent < 70:
                    st.warning("""
                    **⚠️ Moderate Confidence**
                    This might be a difficult case (small drone, complex background)
                    """)

if __name__ == "__main__":
    main()