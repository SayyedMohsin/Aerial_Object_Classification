import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import os

# Page config
st.set_page_config(
    page_title="Aerial Object Classification",
    page_icon="🛸",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-title {
        text-align: center;
        color: #1f77b4;
        font-size: 3rem;
        margin-bottom: 2rem;
    }
    .prediction-card {
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .bird-card {
        background: linear-gradient(135deg, #d4edda, #c3e6cb);
        border: 2px solid #28a745;
    }
    .drone-card {
        background: linear-gradient(135deg, #f8d7da, #f5c6cb);
        border: 2px solid #dc3545;
    }
    .confidence-high { color: #28a745; font-weight: bold; font-size: 2rem; }
    .confidence-medium { color: #ffc107; font-weight: bold; font-size: 2rem; }
    .confidence-low { color: #dc3545; font-weight: bold; font-size: 2rem; }
</style>
""", unsafe_allow_html=True)

class PerfectClassifier:
    def __init__(self):
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load the final trained model"""
        model_path = "models/final_model.h5"
        if os.path.exists(model_path):
            try:
                self.model = keras.models.load_model(model_path)
                return True
            except Exception as e:
                st.error(f"❌ Model loading error: {e}")
                return False
        else:
            st.error("❌ Model not found! Please run training first.")
            return False
    
    def preprocess_image(self, image):
        """Preprocess image exactly like training"""
        # Resize to 150x150 (same as training)
        image = image.resize((150, 150))
        
        # Convert to array and normalize
        img_array = np.array(image) / 255.0
        
        # Handle RGBA images
        if len(img_array.shape) == 3 and img_array.shape[-1] == 4:
            img_array = img_array[:, :, :3]
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def predict(self, image):
        """Make prediction with CORRECT class mapping"""
        if self.model is None:
            return None, None
        
        try:
            processed_image = self.preprocess_image(image)
            prediction = self.model.predict(processed_image, verbose=0)
            raw_score = float(prediction[0][0])
            
            # 🎯 CORRECT CLASS MAPPING (based on training):
            # bird = 0, drone = 1
            # So: score > 0.5 = Drone, score < 0.5 = Bird
            if raw_score > 0.5:
                class_name = "Drone"
                confidence = raw_score
            else:
                class_name = "Bird"
                confidence = 1 - raw_score
            
            return class_name, confidence
            
        except Exception as e:
            st.error(f"Prediction error: {e}")
            return None, None

def main():
    classifier = PerfectClassifier()
    
    # Header
    st.markdown('<h1 class="main-title">🛸 Aerial Object Classification</h1>', unsafe_allow_html=True)
    st.markdown("### Classify images as **Bird** or **Drone** with high accuracy")
    
    # Sidebar
    st.sidebar.title("Model Information")
    st.sidebar.success("""
    **Model Details:**
    - ✅ Correct Class Labels
    - ✅ 150x150 Input Size  
    - ✅ High Accuracy
    - ✅ Proper Preprocessing
    """)
    
    st.sidebar.info("""
    **Class Mapping:**
    - 🐦 Bird = 0
    - 🚁 Drone = 1
    """)
    
    # File upload
    st.header("📷 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an image file", 
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear image of a bird or drone"
    )
    
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📤 Uploaded Image")
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption=f"Size: {image.size}", use_column_width=True)
            except Exception as e:
                st.error(f"Error loading image: {e}")
                return
        
        with col2:
            st.subheader("🎯 Prediction Results")
            
            if not classifier.model:
                st.error("Model not loaded. Please check if 'models/final_model.h5' exists.")
                return
            
            # Make prediction
            with st.spinner("🔍 Analyzing image..."):
                class_name, confidence = classifier.predict(image)
            
            if class_name:
                confidence_percent = confidence * 100
                
                # Determine confidence level
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
                
                # Display prediction card
                if class_name == "Bird":
                    st.markdown(f"""
                    <div class="prediction-card bird-card">
                        <h2>🐦 Prediction: BIRD</h2>
                        <h3 class="{conf_class}">{confidence_percent:.2f}%</h3>
                        <p>{conf_text}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.success("""
                    **Bird Characteristics:**
                    - Organic, curved shapes
                    - Feather textures visible
                    - Natural wing formations
                    - Smooth, flowing contours
                    """)
                    
                else:  # Drone
                    st.markdown(f"""
                    <div class="prediction-card drone-card">
                        <h2>🚁 Prediction: DRONE</h2>
                        <h3 class="{conf_class}">{confidence_percent:.2f}%</h3>
                        <p>{conf_text}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.info("""
                    **Drone Characteristics:**
                    - Mechanical, angular structure
                    - Visible propellers
                    - Straight edges and corners
                    - Man-made components
                    """)
                
                # Debug information
                with st.expander("🔧 Technical Details"):
                    st.write(f"**Model:** final_model.h5")
                    st.write(f"**Input Size:** 150x150 pixels")
                    st.write(f"**Class Mapping:** Bird=0, Drone=1")
                    st.write(f"**Confidence Threshold:** >0.5 = Drone, <0.5 = Bird")
                    
            else:
                st.error("❌ Prediction failed! Please try another image.")
    
    else:
        # Instructions
        st.info("""
        **📋 Instructions:**
        1. Click **'Browse files'** to upload an image
        2. Supported formats: JPG, JPEG, PNG
        3. Image will be automatically resized to 150x150
        4. Model will predict with confidence score
        
        **💡 Tips for best results:**
        - Use clear, well-lit images
        - Center the object in the frame
        - Avoid blurry or dark images
        """)

if __name__ == "__main__":
    main()