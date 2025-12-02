import streamlit as st
import numpy as np
from PIL import Image
import os
import sys

# Add current directory to path to import ensemble_model
sys.path.append(os.path.dirname(__file__))

try:
    from ensemble_model import EnsembleClassifier
    IMPORT_SUCCESS = True
except ImportError as e:
    st.error(f"❌ Could not import EnsembleClassifier: {e}")
    IMPORT_SUCCESS = False

st.set_page_config(
    page_title="Aerial Classification - ENSEMBLE PRO",
    page_icon="🤖", 
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .ensemble-header {
        text-align: center;
        color: #1f77b4;
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    .model-card {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 5px solid;
    }
    .cnn-card {
        border-left-color: #28a745;
        background: #f8fff9;
    }
    .transfer-card {
        border-left-color: #17a2b8;
        background: #f8fcff;
    }
    .ensemble-card {
        border-left-color: #ffc107;
        background: #fffdf6;
    }
    .confidence-high { color: #28a745; font-weight: bold; font-size: 2rem; }
    .confidence-medium { color: #ffc107; font-weight: bold; font-size: 2rem; }
    .confidence-low { color: #dc3545; font-weight: bold; font-size: 2rem; }
</style>
""", unsafe_allow_html=True)

def main():
    st.markdown('<h1 class="ensemble-header">🤖 Advanced Ensemble Classification</h1>', unsafe_allow_html=True)
    st.write("**Combining CNN + Transfer Learning for Maximum Accuracy**")
    
    if not IMPORT_SUCCESS:
        st.error("""
        ❌ Ensemble module not found!
        Please ensure `ensemble_model.py` exists in the project directory.
        """)
        return
    
    # Initialize classifier
    classifier = EnsembleClassifier()
    
    # Display model status
    st.sidebar.title("🔧 Model Status")
    if classifier.models:
        st.sidebar.success(f"✅ {len(classifier.models)} Models Loaded")
        for model_name in classifier.models.keys():
            st.sidebar.info(f"**{model_name.upper()}** Model")
    else:
        st.sidebar.error("❌ No models loaded")
        st.sidebar.info("""
        Required models:
        - models/final_model.h5 (CNN)
        - models/transfer_model.h5 (Transfer Learning)
        """)
    
    # File upload
    st.header("📤 Upload Difficult Image")
    st.write("**Test with small drones, complex backgrounds, or challenging cases**")
    
    uploaded_file = st.file_uploader(
        "Choose image file", 
        type=['jpg', 'jpeg', 'png'],
        help="Upload images that were previously misclassified"
    )
    
    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📷 Input Image")
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption=f"Size: {image.size}", use_column_width=True)
                
                # Image analysis
                st.info(f"""
                **Image Analysis:**
                - Dimensions: {image.size}
                - Format: {image.format}
                - Mode: {image.mode}
                """)
                
            except Exception as e:
                st.error(f"❌ Error loading image: {e}")
                return
        
        with col2:
            st.subheader("🎯 Ensemble Prediction")
            
            if not classifier.models:
                st.error("No models available for prediction!")
                return
            
            # Make prediction
            with st.spinner("🤖 Multiple AI models analyzing..."):
                class_name, confidence, raw_scores = classifier.predict_ensemble(image)
            
            if class_name:
                confidence_percent = confidence * 100
                
                # Confidence level
                if confidence_percent >= 85:
                    conf_class = "confidence-high"
                    conf_text = "Very High Confidence"
                elif confidence_percent >= 70:
                    conf_class = "confidence-medium" 
                    conf_text = "High Confidence"
                elif confidence_percent >= 60:
                    conf_class = "confidence-medium"
                    conf_text = "Good Confidence"
                else:
                    conf_class = "confidence-low"
                    conf_text = "Low Confidence"
                
                # Display ensemble result
                if class_name == "BIRD":
                    st.success(f"## 🐦 FINAL PREDICTION: BIRD")
                else:
                    st.success(f"## 🚁 FINAL PREDICTION: DRONE")
                
                st.markdown(f'<div class="{conf_class}">{confidence_percent:.2f}%</div>', unsafe_allow_html=True)
                st.write(f"**{conf_text}**")
                
                # Individual model results
                st.subheader("🔍 Model Breakdown")
                
                if 'cnn' in classifier.models:
                    st.markdown('<div class="model-card cnn-card">', unsafe_allow_html=True)
                    st.write("**🧠 CNN Model** (150x150)")
                    st.write("Best for: Standard cases, clear images")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                if 'transfer' in classifier.models:
                    st.markdown('<div class="model-card transfer-card">', unsafe_allow_html=True)
                    st.write("**🚀 Transfer Learning Model** (224x224)")
                    st.write("Best for: Small objects, complex backgrounds")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Raw scores
                if raw_scores:
                    st.info("**Raw Model Scores:**")
                    for score in raw_scores:
                        st.write(f"- {score}")
                
                # Special handling for difficult cases
                if confidence_percent < 70:
                    st.warning("""
                    **⚠️ Challenging Case Detected**
                    This image might contain:
                    - Very small drone
                    - Complex background (grass, trees)
                    - Low resolution
                    - Unusual angle
                    """)
                
                st.balloons()
                
            else:
                st.error("❌ Prediction failed! Please try another image.")

if __name__ == "__main__":
    main()