import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Page configuration
st.set_page_config(
    page_title="Aerial Object Classification - PERFECT",
    page_icon="🛸",
    layout="wide"
)

# Custom CSS for beautiful design
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        font-size: 3.5rem;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.5rem;
        margin-bottom: 2rem;
    }
    .prediction-card {
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        text-align: center;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        border: 3px solid;
        transition: all 0.3s ease;
    }
    .bird-card {
        background: linear-gradient(135deg, #d4edda, #c3e6cb);
        border-color: #28a745;
    }
    .drone-card {
        background: linear-gradient(135deg, #f8d7da, #f5c6cb);
        border-color: #dc3545;
    }
    .confidence-high {
        color: #28a745;
        font-weight: bold;
        font-size: 2.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .confidence-medium {
        color: #ffc107;
        font-weight: bold;
        font-size: 2.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .confidence-low {
        color: #dc3545;
        font-weight: bold;
        font-size: 2.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .upload-box {
        border: 2px dashed #1f77b4;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
        background: #f8f9fa;
    }
    .success-box {
        background: #d4edda;
        border: 2px solid #c3e6cb;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background: #d1ecf1;
        border: 2px solid #bee5eb;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class PerfectClassifier:
    def __init__(self):
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model"""
        model_path = "models/final_model.h5"
        if os.path.exists(model_path):
            try:
                self.model = tf.keras.models.load_model(model_path)
                return True
            except Exception as e:
                st.error(f"❌ Model loading error: {e}")
                return False
        else:
            st.error("❌ Model file not found! Please run training first.")
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
        """Make prediction with CORRECT logic"""
        if self.model is None:
            return None, None
        
        try:
            processed_image = self.preprocess_image(image)
            prediction = self.model.predict(processed_image, verbose=0)
            raw_score = float(prediction[0][0])
            
            # 🎯 PERFECT LOGIC (Verified by testing):
            # raw_score < 0.5 = BIRD, raw_score > 0.5 = DRONE
            if raw_score < 0.5:
                class_name = "BIRD"
                confidence = 1 - raw_score  # Convert to confidence for BIRD
            else:
                class_name = "DRONE"
                confidence = raw_score  # Direct score for DRONE
            
            return class_name, confidence, raw_score
            
        except Exception as e:
            st.error(f"Prediction error: {e}")
            return None, None, None

def main():
    # Initialize classifier
    classifier = PerfectClassifier()
    
    # Header Section
    st.markdown('<h1 class="main-header">🛸 Aerial Object Classification</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Bird vs Drone Detection with High Accuracy</p>', unsafe_allow_html=True)
    
    # Sidebar with Model Information
    st.sidebar.title("📊 Model Information")
    st.sidebar.success("""
    **Model Status:** ✅ ACTIVE
    **Accuracy:** 83.26%
    **Precision:** 85.37%
    **Recall:** 74.47%
    """)
    
    st.sidebar.info("""
    **🎯 Classification Logic:**
    - Raw Score < 0.5 → 🐦 BIRD
    - Raw Score > 0.5 → 🚁 DRONE
    """)
    
    st.sidebar.warning("""
    **💡 Tips for Best Results:**
    - Use clear, well-lit images
    - Center the object in frame
    - Avoid blurry images
    - Supported formats: JPG, JPEG, PNG
    """)
    
    # Main Content Area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Image")
        st.markdown('<div class="upload-box">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Drag and drop or click to browse",
            type=['jpg', 'jpeg', 'png'],
            help="Upload an aerial image of a bird or drone",
            key="file_uploader"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption=f"Uploaded Image | Size: {image.size}", use_column_width=True)
                
                # Image information
                st.info(f"""
                **Image Details:**
                - Format: {image.format}
                - Size: {image.size}
                - Mode: {image.mode}
                """)
                
            except Exception as e:
                st.error(f"❌ Error loading image: {e}")
                return
    
    with col2:
        st.header("🎯 Prediction Results")
        
        if uploaded_file is not None:
            if not classifier.model:
                st.error("❌ Model not loaded. Please check if the model file exists.")
                return
            
            # Make prediction
            with st.spinner("🔍 AI is analyzing your image..."):
                class_name, confidence, raw_score = classifier.predict(image)
            
            if class_name:
                confidence_percent = confidence * 100
                
                # Determine confidence level
                if confidence_percent >= 85:
                    conf_class = "confidence-high"
                    conf_emoji = "🟢"
                    conf_text = "Very High Confidence"
                elif confidence_percent >= 70:
                    conf_class = "confidence-medium"
                    conf_emoji = "🟡"
                    conf_text = "High Confidence"
                elif confidence_percent >= 60:
                    conf_class = "confidence-medium"
                    conf_emoji = "🟡"
                    conf_text = "Good Confidence"
                else:
                    conf_class = "confidence-low"
                    conf_emoji = "🔴"
                    conf_text = "Low Confidence"
                
                # Display prediction card
                if class_name == "BIRD":
                    st.markdown(f"""
                    <div class="prediction-card bird-card">
                        <h1>🐦 BIRD DETECTED</h1>
                        <div class="{conf_class}">{conf_emoji} {confidence_percent:.2f}%</div>
                        <h3>{conf_text}</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("""
                    <div class="success-box">
                        <h4>🐦 Bird Characteristics:</h4>
                        <ul>
                        <li>Organic, curved shapes and contours</li>
                        <li>Feather textures and patterns visible</li>
                        <li>Natural wing formations</li>
                        <li>Smooth, flowing body lines</li>
                        <li>Often shows natural flight patterns</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                    
                else:  # DRONE
                    st.markdown(f"""
                    <div class="prediction-card drone-card">
                        <h1>🚁 DRONE DETECTED</h1>
                        <div class="{conf_class}">{conf_emoji} {confidence_percent:.2f}%</div>
                        <h3>{conf_text}</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("""
                    <div class="info-box">
                        <h4>🚁 Drone Characteristics:</h4>
                        <ul>
                        <li>Mechanical, angular structure</li>
                        <li>Visible propellers (usually 4 or more)</li>
                        <li>Straight edges and geometric shapes</li>
                        <li>Man-made components and materials</li>
                        <li>Often shows symmetrical design</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Technical details expander
                with st.expander("🔧 Technical Details"):
                    col_tech1, col_tech2 = st.columns(2)
                    with col_tech1:
                        st.write("**Model Information:**")
                        st.write(f"- Model: final_model.h5")
                        st.write(f"- Input Size: 150×150 pixels")
                        st.write(f"- Framework: TensorFlow")
                        
                    with col_tech2:
                        st.write("**Prediction Details:**")
                        st.write(f"- Raw Score: {raw_score:.4f}")
                        st.write(f"- Confidence: {confidence_percent:.2f}%")
                        st.write(f"- Threshold: 0.5")
                
                # Success message
                st.balloons()
                st.success("🎉 Prediction completed successfully!")
                
            else:
                st.error("❌ Prediction failed! Please try another image.")
        
        else:
            # Instructions when no file uploaded
            st.info("""
            ## 📋 How to Use:
            
            1. **Upload Image**: Click 'Browse files' or drag & drop an image
            2. **Wait for Analysis**: AI will process your image (5-10 seconds)
            3. **View Results**: See prediction with confidence score
            4. **Understand**: Read about the characteristics of detected object
            
            ## 🎯 Supported Images:
            - Clear aerial photographs
            - Birds or drones in flight
            - Well-lit conditions
            - Centered objects work best
            
            **⚠️ Note:** For best accuracy, use images similar to the training data.
            """)

if __name__ == "__main__":
    main()